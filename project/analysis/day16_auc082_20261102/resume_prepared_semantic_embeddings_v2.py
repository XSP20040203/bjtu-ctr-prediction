"""Recover the verified semantic preparation after a terminal CPU predecessor crash."""
from datetime import datetime, timezone
from pathlib import Path
from conditional_daily import read
from fit_stacker import HERE, digest, write


def main():
    folder = HERE / 'runs/full_negative_semantic_embeddings_v1'
    receipt = folder / 'resume_after_predecessor_failure_v2.json'
    assert not receipt.exists() and not (folder / 'models_locked.json').exists()
    assert not list(folder.glob('day*.cbm')), 'Do not duplicate an existing fit'
    failed = HERE / 'runs/full_negative_embedding_lgb_v1/failure.json'
    failure = read(failed)
    assert failure['terminal_exit_code'] == -1073741819
    assert failure['process_id'] == 58516 and failure['native_exception'] == '0xC0000005'
    protocol = read(folder / 'protocol.json')
    preflight = read(folder / 'preflight.json')
    assert preflight['status'] == 'passed'
    assert preflight['protocol_sha256'] == digest(folder / 'protocol.json')
    assert preflight['input_lock_sha256'] == digest(folder / 'inputs_locked.json')
    assert preflight['full_projection_verification_sha256'] == digest(folder / 'full_projection_verification.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    write(receipt, dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        recovery_version=2, preserved_original_queue_session=19435,
        predecessor_session=52215, predecessor_terminal_exit_code=failure['terminal_exit_code'],
        predecessor_failure_sha256=digest(failed),
        reason='Original CPU predecessor terminated in native LightGBM; its wrapper exited and original semantic queue exited1 because predecessor verification was absent. Valid completed semantic preparation is reused under a new recovery launcher.',
        original_semantic_sources_and_protocol_unchanged=True,
        protocol_sha256=digest(folder / 'protocol.json'), preflight_sha256=digest(folder / 'preflight.json'),
        recovery_sources_sha256={str(path): digest(path) for path in (
            Path(__file__), HERE / 'run_full_negative_semantic_embeddings_v2.ps1')},
        inferred17_labels_read=False))
    print({'semantic_recovery_v2':'verified_preparation_reused','columns':489}, flush=True)


if __name__ == '__main__':
    main()
