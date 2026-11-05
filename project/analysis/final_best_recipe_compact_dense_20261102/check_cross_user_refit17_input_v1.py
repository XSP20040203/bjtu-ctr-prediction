"""Replay the final training extension without fitting or opening inferred17 labels."""
from pathlib import Path
import export_cross_user_refit17_v1 as job

HERE=Path(__file__).resolve().parent
OUT=HERE/'cross_user_refit17_input_preflight_v1.json'

if __name__=='__main__':
    assert not OUT.exists(), 'Preserve earlier final-date input checks'
    _,audit=job.training_design(False)
    assert audit['maximum_training_date']==20191116
    assert audit['clicks']==21354
    assert audit['original_prior_rows']==191522
    assert audit['new16_rows']==38578
    job.write(OUT,dict(status='passed',audit=audit,source_sha256=job.digest(job.__file__),
        full_final_training_design_checked=True,model_fitted=False,held_auc_evaluated=False,
        inferred17_labels_read=False))
    job.emit('cross_user_refit17_input_preflight_passed',**audit)
