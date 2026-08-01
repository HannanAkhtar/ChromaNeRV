"""Structural tests for the anonymous NeRV supplementary release."""

import json
from pathlib import Path

import pytest

from nerv_generalization import DEFAULT_CONFIGS, UVG_SEQUENCES
from release_tools import release_files, scan_anonymity
from scripts.aggregate_supplementary_results import aggregate_rows, collect_rows, delta_rows

ROOT = Path(__file__).resolve().parents[1]


def test_configs_describe_complete_grid():
    bunny = json.loads((ROOT / 'configs/supplementary/nerv_bunny.json').read_text())
    uvg = json.loads((ROOT / 'configs/supplementary/nerv_uvg7.json').read_text())
    assert bunny['sequences'] == ['Bunny']
    assert tuple(uvg['sequences']) == UVG_SEQUENCES
    assert tuple(bunny['configs']) == DEFAULT_CONFIGS
    assert tuple(uvg['configs']) == DEFAULT_CONFIGS
    assert bunny['max_frames'] == uvg['max_frames'] == 132
    assert bunny['checkpoint_policy'] == uvg['checkpoint_policy'] == 'final'


def test_equal_eight_sequence_aggregation_and_matched_delta():
    rows = []
    for index, sequence in enumerate(('Bunny',) + UVG_SEQUENCES, 1):
        rows.extend((
            {'sequence': sequence, 'config_name': 'full_rgb', 'rgb_psnr': index,
             'params_M': 10, 'estimated_gflops': 20},
            {'sequence': sequence, 'config_name': 'chroma_w8', 'rgb_psnr': index + 2,
             'params_M': 8, 'estimated_gflops': 5},
            {'sequence': sequence, 'config_name': 'rgbsplit_w8', 'rgb_psnr': index,
             'params_M': 8, 'estimated_gflops': 5},
        ))
    aggregate = {row['config_name']: row for row in aggregate_rows(rows)}
    assert aggregate['full_rgb']['rgb_psnr'] == pytest.approx(4.5)
    assert aggregate['chroma_w8']['parameter_reduction_percent'] == pytest.approx(20)
    deltas = delta_rows(rows, (('chroma_w8', 'rgbsplit_w8'),))
    rgb_delta = next(row for row in deltas if row['metric'] == 'rgb_psnr')
    assert rgb_delta['delta'] == pytest.approx(2)


def test_release_allowlist_and_anonymity():
    files = release_files(ROOT)
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    assert 'README.md' in relative
    assert all(not name.startswith(('.git/', 'data/', 'output/')) for name in relative)
    assert not scan_anonymity(ROOT, files)


def test_aggregation_detects_missing_and_duplicate_runs(tmp_path):
    rows, missing = collect_rows(tmp_path / 'bunny', tmp_path / 'uvg')
    assert not rows
    assert len(missing) == 48
    first = tmp_path / 'bunny/Bunny/full_rgb'
    duplicate = tmp_path / 'bunny/full_rgb'
    first.mkdir(parents=True)
    duplicate.mkdir(parents=True)
    (first / 'eval_metrics.json').write_text('{}')
    (duplicate / 'eval_metrics.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Duplicate result files'):
        collect_rows(tmp_path / 'bunny', tmp_path / 'uvg')


def test_incomplete_reference_results_are_not_presented_as_measurements():
    manifest = json.loads((ROOT / 'results/supplementary/manifest.json').read_text())
    assert manifest['complete'] is False
    assert manifest['included_jobs'] == 0
    assert manifest['expected_jobs'] == 48
