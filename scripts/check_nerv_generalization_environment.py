#!/usr/bin/env python
import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metrics.main_paper_metrics import ffmpeg_libvmaf_info  # noqa: E402
from nerv_generalization import environment_info  # noqa: E402


def import_status(module):
    try:
        imported = importlib.import_module(module)
        return {'available': True, 'version': getattr(imported, '__version__', 'unknown')}
    except Exception as exc:
        return {'available': False, 'reason': str(exc)}


def main():
    report = environment_info(REPO_ROOT)
    for label, module in (
            ('lpips', 'lpips'),
            ('dists', 'DISTS_pytorch'),
            ('torchmetrics_fid', 'torchmetrics.image.fid'),
            ('torch_fidelity', 'torch_fidelity')):
        report[label] = import_status(module)
    try:
        report['vmaf'] = {'available': True, **ffmpeg_libvmaf_info()}
    except Exception as exc:
        report['vmaf'] = {'available': False, 'reason': str(exc)}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

