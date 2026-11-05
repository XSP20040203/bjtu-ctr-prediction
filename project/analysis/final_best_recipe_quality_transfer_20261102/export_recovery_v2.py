"""Repair the exporter ID allowlist while preserving its frozen model recipe."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import export_best as frozen
from group_temperature import FIELDS, group_codes as checked_group_codes

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE/'run_v1'
RUN = HERE/'run_v2'
frozen.RUN = RUN


def declared_group_codes(frame, fields):
    """The checked transform receives exactly its four allowed unlabeled IDs."""
    if not set(FIELDS).issubset(frame.columns):
        raise ValueError('All four declared group IDs must be present')
    return checked_group_codes(frame[FIELDS], fields)


frozen.group_codes = declared_group_codes


def initialize():
    assert not RUN.exists() and not frozen.OUTPUT.exists(), 'Preserve prior runs and exports'
    protocol = frozen.read(PREVIOUS/'protocol.json')
    assert protocol['sources_sha256'][str(HERE/'export_best.py')] == frozen.digest(HERE/'export_best.py')
    paths = [Path(__file__), PREVIOUS/'protocol.json']
    for day in [16, 17]:
        directory = PREVIOUS/f'input_day{day}'
        contract = frozen.read(directory/'contract.json')
        assert frozen.digest(directory/'features.npy') == contract['matrix_sha256']
        assert frozen.digest(directory/'daily_quality.npz') == contract['quality_file_sha256']
        paths += [directory/name for name in ['features.npy', 'daily_quality.npz', 'contract.json']]
    failure_path = PREVIOUS/'failure_record.json'
    assert not failure_path.exists()
    frozen.write(failure_path, dict(phase='predict', dataset=20191116,
        error='Group transforms accept only the four declared unlabeled IDs',
        cause='Exporter passed pt_d alongside the four permitted group IDs',
        first_three_day16_model_forecasts_replayed=True,
        day17_submission_or_inferred_label_evaluation_completed=False,
        frozen_source_preserved=True, recovery_source=str(Path(__file__))))
    paths.append(failure_path)
    RUN.mkdir()
    # Read-only cached inputs have identical bytes; hardlinks avoid duplicating2.5GB.
    for day in [16, 17]:
        destination = RUN/f'input_day{day}'
        destination.mkdir()
        for name in ['features.npy', 'daily_quality.npz', 'contract.json']:
            os.link(PREVIOUS/f'input_day{day}'/name, destination/name)
    protocol.update(status='frozen_recovery_before_day17_prediction_and_inferred_label_evaluation',
        recovery_frozen_utc=datetime.now(timezone.utc).isoformat(),
        recovery='Project to the four allowed IDs before the existing checked group transform',
        inherited_recipe_protocol_sha256=frozen.digest(PREVIOUS/'protocol.json'),
        no_model_or_blend_recipe_changes=True)
    protocol['sources_sha256'].update({str(path):frozen.digest(path) for path in paths})
    frozen.write(RUN/'protocol.json', protocol)
    frozen.emit('recovery_frozen', development16_auc=protocol['development16_auc'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['initialize', 'predict', 'rebuild17', 'verify'])
    args = parser.parse_args()
    if args.phase == 'initialize':
        initialize()
    elif args.phase == 'predict':
        frozen.predict()
    elif args.phase == 'rebuild17':
        frozen.build(20191117, verifying=True)
    else:
        frozen.verify()
