"""Replay the frozen final stage only. This script never reads advertisement labels."""
from pathlib import Path
import argparse,hashlib,json,tempfile,shutil
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from catboost import CatBoost,Pool

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True,help='Restored 大作业上传 folder')
    ap.add_argument('--output',type=Path,required=True,help='A new output folder, never the original submission folder')
    a=ap.parse_args();root=a.project.resolve()
    config=json.loads((root/'best_replay_inputs/replay_config.json').read_text(encoding='utf-8'))
    assert config['selected']['mode']=='replace_global' and config['selected']['weight']==0.1
    features=np.load(root/config['features'],mmap_mode='r')
    mask=np.load(root/config['mask'])
    assert digest(root/config['features'])==config['feature_file_sha256']
    assert digest(root/config['model'])==config['model_sha256']
    assert digest(root/config['reference_csv'])==config['submission_sha256']
    assert hashlib.sha256(mask.tobytes()).hexdigest()==config['input_contract']['selected_mask_sha256']
    model=CatBoost()
    with tempfile.TemporaryDirectory(prefix='ctr_best_model_') as temporary:
        model_path=Path(temporary)/'model.cbm'
        shutil.copyfile(root/config['model'],model_path)
        model.load_model(str(model_path))
    assert model.feature_names_==config['columns']
    margin=np.empty(len(features),dtype='float64');h=hashlib.sha256()
    for start in range(0,len(features),10000):
        stop=min(start+10000,len(features))
        frame=pd.DataFrame(np.array(features[start:stop],copy=True),columns=config['columns'])
        for name in config['categories']:frame[name]=frame[name].to_numpy(dtype='int64')
        h.update(frame.to_numpy(dtype='float32').tobytes(order='C'))
        margin[start:stop]=model.predict(Pool(frame,cat_features=config['categories']),prediction_type='RawFormulaVal',thread_count=2)
    assert h.hexdigest()==config['input_contract']['row_major_float32_frame_sha256']
    expected_margin=np.load(root/config['reference_margin'])
    np.testing.assert_array_equal(margin,expected_margin)
    assert hashlib.sha256(margin.tobytes()).hexdigest()==config['input_contract']['margin_sha256']
    parent=np.load(root/config['parent'])
    z=logit(np.clip(parent,1e-8,1-1e-8))
    normalized=(margin-margin.mean())/max(margin.std(),1e-6)*z.std()
    delta=normalized+z.mean()-z
    prediction=np.where(mask,expit(z+config['selected']['weight']*delta),parent)
    reference=np.load(root/config['reference_prediction'])
    np.testing.assert_allclose(prediction,reference,rtol=0,atol=2e-13)
    a.output.mkdir(parents=True,exist_ok=False)
    csv=a.output/'submission_best_20191117.csv'
    pd.DataFrame({'id':np.arange(len(prediction)),'pctr':prediction}).to_csv(csv,index=False,float_format='%.17g')
    assert digest(csv)==config['submission_sha256']
    report={'status':'passed','scope':config['scope'],'rows':len(prediction),'features':features.shape[1],
            'maximum_margin_error':float(np.max(np.abs(margin-expected_margin))),
            'maximum_prediction_error':float(np.max(np.abs(prediction-reference))),
            'submission_sha256':digest(csv),'day17_advertisement_labels_read':False}
    (a.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
