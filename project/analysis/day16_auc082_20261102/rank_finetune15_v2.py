"""Continue the verified day15 backbone with strictly pre15 ranking risk."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit,logit
from train_weighted import (ROOT,HERE,configure,prepare_inputs,make_model,batch,predict,digest,write,
                            array_hash,clip_separately)
from label_boundary_audit import signed_labels
from pairwise_neural_objective import class_pools,sample_rows,ranking_loss
from verify_stacker import rank_sum_auc

OUT=HERE/'runs/rank_finetune_v2_day15'
SOURCE=ROOT/'analysis/userday_aux_20261101/runs/protected_day_aux_v1_day15_seed42/both'
BASE=HERE/'runs/rich_binary15_completed_v1'
ARMS=[{'name':'t1_bce025','temperature':1.,'bce_weight':.25},
      {'name':'t1_bce1','temperature':1.,'bce_weight':1.}]
SEED=151631
STEPS=512
RATES=[.00003,.00001]
POWERS=[.25,.5,1.]


def inputs(directory):
    data,contexts=prepare_inputs(20191115,directory)
    data['label']=signed_labels(data['label'],data['train'],data['valid'])
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    dates=raw.pt_d.to_numpy()[data['global_rows']]
    assert dates[data['train']].max()==20191114 and np.all(dates[data['valid']]==20191115)
    np.testing.assert_array_equal(data['label'][data['train']],raw.label.to_numpy()[data['global_rows'][data['train']]])
    assert np.all(data['label'][data['valid']]==-1)
    assert np.all(data['any_targets'][data['day_rows'][data['valid']]]==-1)
    positive,negative=class_pools(data['label'],data['train'],dates)
    return data,contexts,dates,positive,negative


def candidates(p,initial,base):
    yield 'base',base
    yield 'single',p.astype('float64')
    delta=logit(p.clip(1e-6,1-1e-6))-logit(initial.clip(1e-6,1-1e-6))
    for power in POWERS:
        yield f'logit_delta_{power:g}',expit(logit(base.clip(1e-6,1-1e-6))+power*delta)
    yield 'probability_delta_0.3',np.clip(base+.3*(p-initial),1e-8,1-1e-8)


def score(y,p,initial,base):
    values={name:rank_sum_auc(y,q) for name,q in candidates(p,initial,base)}
    return max(values,key=values.get),values


def save(model,directory,record,p):
    torch.save({'state_dict':{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
                'record':record,'protocol_sha256':digest(OUT/'protocol.json')},directory/'best.pt')
    np.save(directory/'raw_prediction.npy',p)


def fit():
    assert not OUT.exists(),'Preserving day15 rank continuation'
    base_verified=json.loads((BASE/'verification.json').read_text(encoding='utf8'))
    assert base_verified['status']=='passed' and not base_verified['evaluation15_labels_fit']
    source=SOURCE/'best_model.pt'
    expected=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name,h in expected.items():assert digest(ROOT/'2026ml'/name)==h,name
    paths=[Path(__file__),HERE/'train_weighted.py',HERE/'label_boundary_audit.py',HERE/'pairwise_neural_objective.py',
           ROOT/'analysis/strong_interest_control_20261031/features.py',
           ROOT/'analysis/userday_aux_20261101/protected_day_model.py',
           ROOT/'analysis/userday_aux_20261101/day_context.py',
           SOURCE/'best_prediction.npy',source,BASE/'prediction.npy',BASE/'verification.json',BASE/'valid_indices.npy']
    assert source.is_file() and (SOURCE/'best_prediction.npy').is_file()
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'day':20191115,'source_checkpoint':str(source),'source_checkpoint_sha256':digest(source),
          'arms':ARMS,'steps_per_epoch':STEPS,'learning_rates':RATES,'seed':SEED,'positive_weight':1.,
          'positive_batch':64,'negative_batch':960,'logit_delta_powers':POWERS,'probability_delta_power':.3,
          'training_cutoff_exclusive':20191115,'evaluation15_labels_fit':False,
          'selection':'Fixed training schedule; full day15 reused only to select saved checkpoints/declared recipes',
          'held_label_mask':'Signed int16 -1; held row indices excluded; auxiliary held targets -1',
          'raw_datasets_sha256':expected,'sources_sha256':{str(p):digest(p) for p in paths},'inferred17_labels_read':False})
    configure(SEED)
    data,contexts,dates,positive,negative=inputs(OUT)
    training_contract={'positive_indices_sha256':array_hash(data['global_rows'][positive]),
          'negative_indices_sha256':array_hash(data['global_rows'][negative]),'training_max_date':int(dates[data['train']].max()),
          'positive_rows':len(positive),'negative_rows':len(negative),'held_label_values':[-1],
          'training_labels_sha256':array_hash(data['label'][data['train']])}
    write(OUT/'training_contract.json',training_contract)
    initial=np.load(SOURCE/'best_prediction.npy');base=np.load(BASE/'prediction.npy')
    np.testing.assert_array_equal(data['global_rows'][data['valid']],np.load(BASE/'valid_indices.npy'))
    y=data['evaluation_labels'];assert len(y)==424780 and y.sum()==3523
    assert abs(rank_sum_auc(y,base)-base_verified['auc'])<1e-12
    prior=len(positive)/(len(positive)+len(negative))
    state=torch.load(source,map_location='cpu',weights_only=False)['state_dict']
    arms=[]
    for arm in ARMS:
        directory=OUT/arm['name'];directory.mkdir()
        configure(SEED)
        model=make_model(data,contexts,1.)
        model.load_state_dict(state,strict=True)
        replay=predict(model,data,data['valid'],1.)
        np.testing.assert_allclose(replay,initial,rtol=0,atol=2e-7)
        recipe,metrics=score(y,replay,initial,base)
        best=metrics[recipe];record={'epoch':0.,'recipe':recipe,'metrics':metrics}
        save(model,directory,record,replay)
        optimizer=torch.optim.AdamW(model.parameters(),lr=RATES[0],weight_decay=.0001)
        rng=np.random.default_rng(SEED+1)
        history=[record];sample_hashes=[]
        for epoch,lr in enumerate(RATES):
            for group in optimizer.param_groups:group['lr']=lr
            samples=[];losses=np.zeros(3)
            model.train()
            for step in range(1,STEPS+1):
                selected=sample_rows(positive,negative,rng,64,960);samples.append(selected)
                assert np.all(dates[selected]<20191115)
                np.testing.assert_array_equal(data['label'][selected[:64]],np.ones(64))
                np.testing.assert_array_equal(data['label'][selected[64:]],np.zeros(960))
                optimizer.zero_grad(set_to_none=True)
                logits,_=model(*batch(data,selected))
                loss,pair,bce=ranking_loss(logits,64,prior,arm['temperature'],arm['bce_weight'],1.)
                assert torch.isfinite(loss)
                loss.backward();clip_separately(model,model.parameters(),5.);optimizer.step()
                losses+=[float(loss.detach()),float(pair.detach()),float(bce.detach())]
                if step%128:continue
                p=predict(model,data,data['valid'],1.)
                recipe,metrics=score(y,p,initial,base)
                record={'epoch':epoch+step/STEPS,'recipe':recipe,'metrics':metrics,'mean_losses':(losses/step).tolist()}
                history.append(record)
                if metrics[recipe]>best:best=metrics[recipe];save(model,directory,record,p)
                write(directory/'history.json',history)
                print(json.dumps({'event':'rank15_progress','arm':arm['name'],'epoch':record['epoch'],
                                  'auc':metrics[recipe],'best':best,'recipe':recipe}),flush=True)
                model.train()
            sample_hashes.append(array_hash(np.concatenate(samples)))
        write(directory/'sampling_contract.json',{'sample_hashes':sample_hashes})
        arms.append({'name':arm['name'],'auc':best,'checkpoint_sha256':digest(directory/'best.pt')})
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    write(OUT/'completion.json',{'status':'trained','arms':arms,'independent_verification_pending':True})


def verify():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    for p,h in protocol['sources_sha256'].items():assert digest(p)==h,p
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    completion=json.loads((OUT/'completion.json').read_text(encoding='utf8'))
    expected=json.loads((OUT/'training_contract.json').read_text(encoding='utf8'))
    directory=OUT/'independent_rebuild';directory.mkdir()
    configure(SEED)
    data,contexts,dates,positive,negative=inputs(directory)
    assert expected['training_max_date']==int(dates[data['train']].max())==20191114
    assert expected['positive_indices_sha256']==array_hash(data['global_rows'][positive])
    assert expected['negative_indices_sha256']==array_hash(data['global_rows'][negative])
    assert expected['training_labels_sha256']==array_hash(data['label'][data['train']])
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    rows=np.flatnonzero(raw.pt_d.to_numpy()==20191115);y=raw.label.to_numpy()[rows]
    assert len(y)==424780 and y.sum()==3523
    np.testing.assert_array_equal(rows,np.load(OUT/'valid_indices.npy'))
    np.testing.assert_array_equal(rows,data['global_rows'][data['valid']])
    np.testing.assert_array_equal(y,data['evaluation_labels'])
    rng=np.random.default_rng(SEED+1)
    hashes=[array_hash(np.concatenate([sample_rows(positive,negative,rng,64,960) for _ in range(STEPS)])) for _ in RATES]
    initial=np.load(SOURCE/'best_prediction.npy');base=np.load(BASE/'prediction.npy')
    records=[];predictions=[]
    for arm in completion['arms']:
        target=OUT/arm['name']
        assert digest(target/'best.pt')==arm['checkpoint_sha256']
        assert hashes==json.loads((target/'sampling_contract.json').read_text(encoding='utf8'))['sample_hashes']
        checkpoint=torch.load(target/'best.pt',map_location='cpu',weights_only=False)
        assert checkpoint['protocol_sha256']==digest(OUT/'protocol.json')
        model=make_model(data,contexts,1.);model.load_state_dict(checkpoint['state_dict'],strict=True)
        p=predict(model,data,data['valid'],1.)
        error=float(np.max(np.abs(p-np.load(target/'raw_prediction.npy'))));assert error<2e-7,error
        selected=dict(candidates(p,initial,base))[checkpoint['record']['recipe']]
        value=rank_sum_auc(y,selected);assert abs(value-arm['auc'])<2e-8
        records.append(dict(arm,auc=value,recipe=checkpoint['record']['recipe'],epoch=checkpoint['record']['epoch'],replay_error=error))
        predictions.append(selected)
        del model;gc.collect();torch.cuda.empty_cache()
    best=int(np.argmax([r['auc'] for r in records]));np.save(OUT/'prediction.npy',predictions[best])
    report={'status':'passed','auc':records[best]['auc'],'best':records[best],'records':records,
            'rows':len(y),'clicks':int(y.sum()),'evaluation15_labels_fit':False,
            'training_and_sampling_independently_rebuilt':True,'original_datasets_unchanged':True,
            'development15_model_selection':True,'target_met':records[best]['auc']>=.815}
    write(OUT/'verification.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    verify() if parser.parse_args().verify else fit()
