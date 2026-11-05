"""Preserve the locked v1 source and fix its cache row-alignment assumption."""
from pathlib import Path
import hashlib

here=Path(__file__).resolve().parent
source=here/'predict_and_verify.py'
target=here/'predict_and_verify_v2.py'
assert not target.exists()
text=source.read_text(encoding='utf8')
text=text.replace("RUN = HERE/'run_v1'", "RUN = HERE/'run_v2'")
old="""    # Feature reconstruction must reproduce the already verified day16 input.
    original = np.load(ROWDIR/'independent_input/features.npy', mmap_mode='r')
    n16 = int(np.sum(dates == 20191116))
    assert n16 == 421983 and int(np.sum(dates == 20191117)) == 404182
    assert digest(ROWDIR/'independent_input/features.npy') == contract['features_sha256']
    for start in range(0,n16,20000):
        stop = min(n16,start+20000)
        np.testing.assert_array_equal(x[start:stop], original[-n16+start:-n16+stop] if stop<n16 else original[-n16+start:])
"""
assert text.count(old)==1
text=text.replace(old,"    assert_day16_features(x, dates, raw)\n")
old="    x,dates,raw = build_features(RUN/'input')\n"
assert text.count(old)==1
text=text.replace(old,"    x,dates,raw = reuse_prepared_input()\n")
helpers='''
def assert_day16_features(x, dates, raw):
    from global_row_context import selected_training
    contract=read(ROWDIR/'input_contract.json')
    original=np.load(ROWDIR/'independent_input/features.npy',mmap_mode='r')
    train,_=selected_training(raw.label.to_numpy(),raw.pt_d.to_numpy())
    held=np.flatnonzero(raw.pt_d.to_numpy()==20191116)
    old_rows=np.sort(np.r_[train,held])
    original_positions=pd.Index(old_rows).get_indexer(held)
    assert np.all(original_positions>=0)
    assert len(held)==421983 and np.sum(dates==20191117)==404182
    assert digest(ROWDIR/'independent_input/features.npy')==contract['features_sha256']
    selected=np.flatnonzero(dates==20191116)
    for start in range(0,len(held),20000):
        stop=min(len(held),start+20000)
        np.testing.assert_array_equal(x[selected[start:stop]],original[original_positions[start:stop]])
    emit('day16_feature_alignment_verified',rows=len(held))


def reuse_prepared_input():
    directory=HERE/'run_v1/input'
    train=pd.read_csv(ROOT/'2026ml/train.csv',usecols=CAT+['pt_d','label'],dtype='int32')
    final=pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=CAT+['pt_d'],dtype='int32')
    final['label']=-1
    raw=pd.concat([train,final],ignore_index=True)
    rows=np.flatnonzero(raw.pt_d.to_numpy()>=20191116)
    dates=raw.pt_d.to_numpy()[rows]
    x=np.load(directory/'features.npy',mmap_mode='r')
    assert x.shape==(len(rows),len(read(ROWDIR/'input_contract.json')['feature_names']))
    assert_day16_features(x,dates,raw)
    for start in range(0,len(rows),20000):
        np.testing.assert_array_equal(x[start:start+20000,:len(CAT)],raw.iloc[rows[start:start+20000]][CAT].to_numpy())
        assert np.isfinite(x[start:start+20000]).all()
    write(RUN/'reused_input.json',{'path':str(directory/'features.npy'),'sha256':digest(directory/'features.npy'),
          'reason':'v1 generated features fully; v2 corrects chronological/cache alignment check',
          'day16_exact_feature_replay':True})
    return x,dates,raw


'''
marker="if __name__=='__main__':"
assert text.count(marker)==1
text=text.replace(marker,helpers+marker)
target.write_text(text,encoding='utf8')
print({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'new_source':str(target)})
