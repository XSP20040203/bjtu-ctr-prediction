"""Complete-quality daily residual networks; all model fits precede held scoring."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from quality_daily_catboost import inputs as native_inputs, training_targets
from bayesian_daily_offset import offsets
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash, correct
from verify_stacker import rank_sum_auc

SOURCE = HERE/'runs/quality_daily_catboost_v1'
BASE = HERE/'runs/compose_quality_daily_transfer_v1'
OUT = HERE/'runs/quality_daily_neural_v1'
QUEUE = HERE/'quality_daily_neural_queued_v1.json'
CONFIG = dict(seeds=[42,2026],checkpoints=[3,8,16,32],width=128,embedding_dim=8,
    input_dropout=.1,dropout=.2,residual_bound=4.,batch_size=256,prediction_batch_size=1024,
    learning_rate=.001,weight_decay=.03,count_loss_weight=.25,powers=[.0625,.125,.25,.5])


def normalization(frame,keys,categories,cutoff):
    train = np.flatnonzero(np.asarray(keys)[:,0] < cutoff)
    if not len(train):
        raise ValueError('Normalization requires strictly earlier groups')
    columns = [name for name in frame if name not in categories]
    center,scale = [],[]
    for first in range(0,len(columns),64):
        values = frame[columns[first:first+64]].iloc[train].to_numpy(dtype='float64')
        center.extend(values.mean(axis=0))
        scale.extend(values.std(axis=0).clip(.05))
    result = dict(center=np.asarray(center,dtype='float32'),scale=np.asarray(scale,dtype='float32'))
    for i,name in enumerate(categories):
        result[f'vocab_{i}'] = np.unique(frame[name].to_numpy(dtype='int64')[train])
    return columns,result


def encode_categories(frame,categories,normal):
    values = []
    for i,name in enumerate(categories):
        vocab = normal[f'vocab_{i}']
        values.append(pd.Index(vocab).get_indexer(frame[name])+1)
    return np.column_stack(values).astype('int64')


class DailyResidualNet(nn.Module):
    def __init__(self,numeric_dim,vocab_sizes):
        super().__init__()
        width = CONFIG['width']
        self.embeddings = nn.ModuleList([nn.Embedding(size,CONFIG['embedding_dim'],padding_idx=0) for size in vocab_sizes])
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight,std=.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.numeric = nn.Sequential(nn.Dropout(CONFIG['input_dropout']),nn.Linear(numeric_dim,width),
            nn.LayerNorm(width),nn.SiLU())
        self.mix = nn.Sequential(nn.Linear(width+len(vocab_sizes)*CONFIG['embedding_dim'],width),
            nn.SiLU(),nn.Dropout(CONFIG['dropout']))
        self.residual = nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.SiLU(),
            nn.Dropout(CONFIG['dropout']),nn.Linear(width,width))
        self.head = nn.Linear(width,2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,numeric,categories,baseline):
        embedded = [embedding(categories[:,i]) for i,embedding in enumerate(self.embeddings)]
        hidden = self.mix(torch.cat([self.numeric(numeric),*embedded],dim=1))
        hidden = hidden+self.residual(hidden)
        bound = CONFIG['residual_bound']
        return baseline+bound*torch.tanh(self.head(hidden)/bound)


def loss_value(margin,target):
    if torch.any(target < 0) or not torch.isfinite(target).all():
        raise ValueError('Only observed earlier group counts may enter training')
    return (nn.functional.binary_cross_entropy_with_logits(margin[:,0],(target>0).float())+
        CONFIG['count_loss_weight']*nn.functional.poisson_nll_loss(margin[:,1].clamp(-12,8),target,log_input=True))


def configure(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build(day,directory,verifying=False):
    directory.mkdir(exist_ok=False)
    data,audit = native_inputs(day)
    assert audit == read(SOURCE/f'input_contract{day}.json')
    train,_ = training_targets(data['keys'],data['counts'],20191100+day,'poisson')
    columns,normal = normalization(data['frame'],data['keys'],data['categories'],20191100+day)
    numeric = np.lib.format.open_memmap(directory/'numeric.npy',mode='w+',dtype='float32',shape=(len(data['keys']),len(columns)))
    for first in range(0,len(columns),64):
        stop = min(first+64,len(columns))
        block = data['frame'][columns[first:stop]].to_numpy(dtype='float32')
        numeric[:,first:stop] = np.clip((block-normal['center'][first:stop])/normal['scale'][first:stop],-8,8)
    numeric.flush()
    categories = encode_categories(data['frame'],data['categories'],normal)
    baseline = np.column_stack([offsets(data,'binary'),offsets(data,'poisson')]).astype('float32')
    np.save(directory/'categories.npy',categories)
    np.savez_compressed(directory/'normalization.npz',**normal)
    meta = dict(keys=data['keys'],counts=data['counts'].astype('float32'),baseline=baseline,
        rows=data['rows'],row_groups=data['row_groups'])
    np.savez_compressed(directory/'meta.npz',**meta)
    contract = dict(native_contract=audit,numeric_names=columns,categories=data['categories'],
        shape=list(numeric.shape),numeric_sha256=digest(directory/'numeric.npy'),
        categories_sha256=array_hash(categories),normalization_sha256={k:array_hash(v) for k,v in normal.items()},
        metadata_sha256={k:array_hash(v) for k,v in meta.items()},
        vocab_sizes=[len(normal[f'vocab_{i}'])+1 for i in range(len(data['categories']))],
        max_normalization_training_date=int(data['keys'][train,0].max()),held_counts_masked=True)
    if verifying:
        assert contract == read(OUT/f'input_day{day}'/'contract.json')
    write(directory/'contract.json',contract)
    del data,numeric,categories,baseline
    gc.collect()
    emit('quality_neural_features_ready',day=day,shape=contract['shape'],verifying=verifying)
    return contract


def queue(gpu_parent,cpu_parent):
    assert not QUEUE.exists() and not OUT.exists(), 'Preserve existing queues and runs'
    paths = [Path(__file__),HERE/'test_quality_daily_neural.py',HERE/'test_quality_daily_catboost.py',
        HERE/'run_quality_daily_neural.ps1',HERE/'quality_daily_catboost.py',
        SOURCE/'verification.json',SOURCE/'protocol.json',BASE/'verification.json',BASE/'protocol.json']
    parent = read(SOURCE/'protocol.json')
    for path,expected in parent['sources_sha256'].items():
        assert digest(path)==expected,path
    tests=subprocess.run([sys.executable,str(HERE/'test_quality_daily_neural.py')],
        capture_output=True,text=True,encoding='utf8',errors='replace',check=True)
    assert 'Ran 6 tests' in tests.stderr and 'OK' in tests.stderr
    paths += [Path(p) for p in parent['sources_sha256']]
    for day in [15,16]:
        paths += [SOURCE/f'input_contract{day}.json',BASE/f'day{day}_prediction.npy',BASE/f'day{day}_indices.npy']
    assert read(SOURCE/'verification.json')['status'] == read(BASE/'verification.json')['status'] == 'passed'
    waits=[]
    for process_id,needle in [(gpu_parent,'run_compact_quality_daily_catboost.ps1'),(cpu_parent,'run_dense_context_rows.ps1')]:
        command=f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(process_id)}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True).stdout.strip()
        if result:
            live=json.loads(result)
            assert needle in live['CommandLine']
        waits.append(dict(process_id=process_id,command_needle=needle,confirmed_live_at_queue=bool(result),
            command_line=live['CommandLine'] if result else None))
    for name,expected in parent['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected,name
    record=dict(config=CONFIG,wait_processes=waits,raw_datasets_sha256=parent['raw_datasets_sha256'],
        sources_sha256={str(path):digest(path) for path in dict.fromkeys(paths)},
        normalization_and_vocab_fit_only_before_prediction_date=True,all337_quality_features=True,
        both_date_models_locked_before_scoring=True,boundary_tests_passed=6,inferred17_labels_read=False)
    write(QUEUE,record)
    emit('quality_neural_queued',waits=waits)


def check_queue():
    queue=read(QUEUE)
    assert queue['config']==CONFIG
    for path,expected in queue['sources_sha256'].items():
        assert digest(path)==expected,path
    for name,expected in queue['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    return queue


def check_sources():
    protocol=read(OUT/'protocol.json')
    assert protocol['queue_sha256']==digest(QUEUE)
    return check_queue()


def tensors(directory):
    numeric=np.load(directory/'numeric.npy',mmap_mode='r')
    result=torch.empty(numeric.shape,dtype=torch.float32,device='cuda')
    for first in range(0,len(numeric),5000):
        result[first:first+5000]=torch.from_numpy(np.array(numeric[first:first+5000],copy=True)).cuda()
    categories=torch.as_tensor(np.load(directory/'categories.npy'),dtype=torch.long,device='cuda')
    meta=np.load(directory/'meta.npz')
    baseline=torch.as_tensor(meta['baseline'],device='cuda')
    counts=torch.as_tensor(meta['counts'],device='cuda')
    return result,categories,baseline,counts,meta


def fit():
    queue=check_queue()
    assert not OUT.exists(), 'Preserve prior fits and failed runs'
    assert torch.cuda.is_available()
    OUT.mkdir()
    write(OUT/'protocol.json',dict(config=CONFIG,queue_sha256=digest(QUEUE),
        dates=[20191115,20191116],raw_datasets_sha256=queue['raw_datasets_sha256'],
        own_held_ad_labels_fit=False,inferred17_labels_read=False,
        selection='Both-date fixed epoch checkpoints lock before held scoring; reused development dates',
        sources_sha256=queue['sources_sha256']))
    locks=[]
    for day in [15,16]:
        directory=OUT/f'input_day{day}'
        contract=build(day,directory)
        x,categories,baseline,counts,meta=tensors(directory)
        training,target=training_targets(meta['keys'],meta['counts'],20191100+day,'poisson')
        assert np.all(meta['keys'][training,0] < 20191100+day)
        configure(42)
        smoke=DailyResidualNet(contract['shape'][1],contract['vocab_sizes']).cuda().eval()
        with torch.no_grad():
            initial=smoke(x[:32],categories[:32],baseline[:32])
            assert torch.equal(initial,baseline[:32])
        del smoke
        emit('quality_neural_full_input_gpu_preflight_passed',day=day,shape=list(x.shape))
        for seed in CONFIG['seeds']:
            configure(seed)
            model=DailyResidualNet(contract['shape'][1],contract['vocab_sizes']).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'],foreach=False)
            rng=np.random.default_rng(seed)
            for epoch in range(1,max(CONFIG['checkpoints'])+1):
                model.train()
                total=0.
                for first in range(0,len(training),CONFIG['batch_size']):
                    if first==0:
                        order=rng.permutation(training)
                    ids=torch.as_tensor(order[first:first+CONFIG['batch_size']],dtype=torch.long,device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    loss=loss_value(model(x[ids],categories[ids],baseline[ids]),counts[ids])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite neural training loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5.)
                    optimizer.step()
                    total+=float(loss.detach())*len(ids)
                if epoch in CONFIG['checkpoints']:
                    name=f'day{day}_seed{seed}_epoch{epoch}'
                    path=OUT/(name+'.pt')
                    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        numeric_dim=contract['shape'][1],vocab_sizes=contract['vocab_sizes']),path)
                    locks.append(dict(name=name,day=day,seed=seed,epoch=epoch,model_sha256=digest(path)))
                    emit('quality_neural_checkpoint_saved',day=day,seed=seed,epoch=epoch,training_loss=total/len(training))
            del model,optimizer
            gc.collect()
        del x,categories,baseline,counts,meta
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/'models_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),models=locks))


def group_forecasts(day,directory,verifying=False):
    configure(42)
    contract=read(directory/'contract.json')
    x,categories,baseline,counts,meta=tensors(directory)
    held=np.flatnonzero(meta['keys'][:,0]==20191100+day)
    assert np.all(meta['counts'][held]==-1)
    locked=read(OUT/'models_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    result={}
    for record in locked['models']:
        if record['day']!=day:
            continue
        path=OUT/(record['name']+'.pt')
        assert digest(path)==record['model_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['numeric_dim']==contract['shape'][1] and saved['vocab_sizes']==contract['vocab_sizes']
        model=DailyResidualNet(saved['numeric_dim'],saved['vocab_sizes']).cuda()
        model.load_state_dict(saved['state_dict'],strict=True)
        model.eval()
        pieces=[]
        with torch.inference_mode():
            for first in range(0,len(held),CONFIG['prediction_batch_size']):
                rows=torch.as_tensor(held[first:first+CONFIG['prediction_batch_size']],dtype=torch.long,device='cuda')
                pieces.append(model(x[rows],categories[rows],baseline[rows]).cpu().numpy())
        margin=np.concatenate(pieces).astype('float64')
        response=np.column_stack([expit(margin[:,0]),np.exp(margin[:,1].clip(-15,8))])
        assert np.isfinite(response).all()
        result[record['name']]=response
        del model,saved
    for epoch in CONFIG['checkpoints']:
        result[f'day{day}_mean_epoch{epoch}']=np.mean(
            [result[f'day{day}_seed{seed}_epoch{epoch}'] for seed in CONFIG['seeds']],axis=0)
    for name,value in result.items():
        path=OUT/(name+'_held_group_prediction.npy')
        if verifying:
            np.testing.assert_array_equal(value,np.load(path))
        else:
            assert not path.exists()
            np.save(path,value)
    metadata={name:meta[name].copy() for name in ['keys','rows','row_groups']}
    del x,categories,baseline,counts,meta
    gc.collect()
    torch.cuda.empty_cache()
    emit('quality_neural_group_forecasts_ready',day=day,forecasts=len(result),verifying=verifying)
    return result,metadata


def independent_correction(base,groups,response,objective,power):
    base=np.clip(np.asarray(base,dtype='float64'),1e-8,1-1e-8)
    if objective=='binary':
        hazard=-np.log1p(-base)
        target=-np.log1p(-np.clip(response,1e-6,1-1e-6))
    elif objective=='poisson':
        hazard=base
        target=np.maximum(response,1e-6)
    else:
        raise ValueError('Undeclared objective')
    table=pd.DataFrame(dict(group=groups,hazard=hazard))
    mass=table.groupby('group',sort=False).hazard.sum().reindex(np.arange(len(response)),fill_value=0).to_numpy()
    ratio=np.clip(target/mass.clip(1e-8),.05,20.)**power
    if objective=='binary':
        return -np.expm1(-hazard*ratio[groups])
    odds=base/(1-base)*ratio[groups]
    return odds/(1+odds)


def candidates(day,forecasts,metadata):
    np.testing.assert_array_equal(metadata['rows'],np.load(BASE/f'day{day}_indices.npy'))
    base=np.load(BASE/f'day{day}_prediction.npy')
    assert len(base)==len(metadata['rows'])=={15:424780,16:421983}[day]
    yield dict(name='base',objective='base',power=0.),base,None
    held=np.flatnonzero(metadata['keys'][:,0]==20191100+day)
    for name,values in forecasts.items():
        assert values.shape==(len(held),2)
        for column,objective in enumerate(['binary','poisson']):
            response=np.zeros(len(metadata['keys']),dtype='float64')
            response[held]=values[:,column]
            for power in CONFIG['powers']:
                record=dict(name=name,objective=objective,power=power)
                p=correct(base,response,metadata['row_groups'],objective,power)
                assert np.isfinite(p).all() and np.all((p>0)&(p<1))
                yield record,p,response


def candidate_key(record):
    return f'{record["name"]}__{record["objective"]}__power{record["power"]:g}'


def score_locked(verifying=False):
    protocol=check_sources()
    locked=read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    actual=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    results={}
    for day in [15,16]:
        directory=OUT/(f'independent_day{day}' if verifying else f'input_day{day}')
        meta=np.load(directory/'meta.npz')
        metadata={name:meta[name] for name in ['keys','rows','row_groups']}
        rows=np.flatnonzero(actual.pt_d.to_numpy()==20191100+day)
        np.testing.assert_array_equal(rows,metadata['rows'])
        y=actual.label.to_numpy()[rows]
        assert int(y.sum())=={15:3523,16:3724}[day]
        forecast_names=locked['dates'][str(day)]['forecasts_sha256']
        forecasts={name:np.load(OUT/(name+'_held_group_prediction.npy')) for name in forecast_names}
        records=[]
        best=None
        winner=None
        maximum_error=0.
        base=np.load(BASE/f'day{day}_prediction.npy')
        for record,p,response in candidates(day,forecasts,metadata):
            assert array_hash(p)==locked['dates'][str(day)]['candidates_sha256'][candidate_key(record)]
            if verifying and response is not None:
                independent=independent_correction(base,metadata['row_groups'],response,record['objective'],record['power'])
                error=float(np.max(np.abs(p-independent)))
                assert error < 2e-14
                maximum_error=max(maximum_error,error)
            record=dict(record,auc=rank_sum_auc(y,p))
            records.append(record)
            if best is None or record['auc']>best['auc']:
                best,winner=record,p.copy()
        assert abs(roc_auc_score(y,winner)-best['auc']) < 1e-12
        if verifying:
            np.testing.assert_array_equal(winner,np.load(OUT/f'day{day}_prediction.npy'))
            assert best==read(OUT/'result.json')['dates'][str(20191100+day)]['best']
        else:
            np.save(OUT/f'day{day}_prediction.npy',winner)
        results[str(20191100+day)]=dict(auc=best['auc'],best=best,records=records,rows=len(y),clicks=int(y.sum()),
            target_met=best['auc']>={15:.815,16:.82}[day],evaluation_day_labels_fit=False,
            held_labels_used_to_select_fixed_recipe=True,all_both_date_fits_locked_before_scoring=True,
            maximum_independent_correction_error=maximum_error,features_rebuilt_from_raw=verifying,
            saved_models_replayed_exactly=verifying,full_cohort_csv_publication_required=True)
        emit('quality_neural_scored',day=day,best=best,verifying=verifying)
    check_sources()
    write(OUT/('verification.json' if verifying else 'result.json'),dict(
        status='passed' if verifying else 'evaluated',dates=results,original_datasets_unchanged=True,
        both_targets_met=all(record['target_met'] for record in results.values())))


def evaluate():
    check_sources()
    assert not (OUT/'predictions_locked.json').exists(), 'Preserve existing locked candidates'
    dates={}
    for day in [15,16]:
        directory=OUT/f'input_day{day}'
        forecasts,metadata=group_forecasts(day,directory)
        np.save(OUT/f'day{day}_indices.npy',metadata['rows'])
        dates[str(day)]=dict(forecasts_sha256={name:array_hash(value) for name,value in forecasts.items()},
            candidates_sha256={candidate_key(record):array_hash(p) for record,p,_ in candidates(day,forecasts,metadata)})
        emit('quality_neural_candidates_locked',day=day,candidates=len(dates[str(day)]['candidates_sha256']))
    write(OUT/'predictions_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        model_lock_sha256=digest(OUT/'models_locked.json'),dates=dates,held_labels_opened=False))
    score_locked(False)


def verify():
    check_sources()
    assert not (OUT/'verification.json').exists(), 'Preserve completed independent verification'
    locked=read(OUT/'predictions_locked.json')
    assert locked['model_lock_sha256']==digest(OUT/'models_locked.json')
    for day in [15,16]:
        directory=OUT/f'independent_day{day}'
        build(day,directory,verifying=True)
        forecasts,metadata=group_forecasts(day,directory,verifying=True)
        for name,value in forecasts.items():
            assert array_hash(value)==locked['dates'][str(day)]['forecasts_sha256'][name]
        for record,p,_ in candidates(day,forecasts,metadata):
            assert array_hash(p)==locked['dates'][str(day)]['candidates_sha256'][candidate_key(record)]
    score_locked(True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['queue','queue-check','fit','evaluate','verify'])
    parser.add_argument('--gpu-parent',type=int,default=110260)
    parser.add_argument('--cpu-parent',type=int,default=101688)
    args=parser.parse_args()
    if args.phase=='queue':
        queue(args.gpu_parent,args.cpu_parent)
    else:
        {'queue-check':check_queue,'fit':fit,'evaluate':evaluate,'verify':verify}[args.phase]()
