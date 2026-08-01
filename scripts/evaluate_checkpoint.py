#!/usr/bin/env python
"""Evaluate one resolved supplementary checkpoint with the central trainer metrics."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metrics.main_paper_metrics import (  # noqa: E402
    compute_pooled_fid, compute_pooled_kid, compute_vmaf)
from nerv_generalization import read_json  # noqa: E402
from persistence import atomic_write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Resolved per-run config.json')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--skip_vmaf', action='store_true')
    parser.add_argument('--vmaf_model_path', default=None)
    parser.add_argument('--vmaf_neg_model_path', default=None)
    parser.add_argument('--compute_fid', action='store_true')
    parser.add_argument('--compute_kid', action='store_true')
    parser.add_argument('--allow_resize', action='store_true')
    return parser.parse_args()


def build_command(args, config):
    required = ('sequence', 'experiment', 'color_space', 'architecture')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            'Standalone evaluation requires a resolved per-run config; missing: '
            + ', '.join(missing))
    architecture = config['architecture']
    width = config.get('branch_width') or architecture['lower_width']
    command = [
        sys.executable, str(REPO_ROOT / 'train_chroma_nerv.py'),
        '--eval_only', '--weight', str(Path(args.checkpoint).resolve()),
        '--job_config_json', str(Path(args.config).resolve()),
        '--data_root', str(Path(args.data_root).resolve()),
        '--sequence', config['sequence'],
        '--max_frames', str(config.get('max_frames', 132)),
        '--experiment', config['experiment'],
        '--color_space', config['color_space'],
        '--embed', architecture['embed'],
        '--stem_dim_num', architecture['stem_dim_num'],
        '--fc_hw_dim', architecture['fc_hw_dim'],
        '--expansion', str(architecture['expansion']),
        '--reduction', str(architecture['reduction']),
        '--lower-width', str(architecture['lower_width']),
        '--num-blocks', str(architecture['num_blocks']),
        '--norm', architecture['norm'], '--act', architecture['act'],
        '--conv_type', architecture['conv_type'],
        '--strides', *map(str, architecture['strides']), '--single_res',
        '--target_height', str(architecture['target_height']),
        '--target_width', str(architecture['target_width']),
        '--y_branch_width', str(width), '--rgb_branch_width', str(width),
        '--lambda_y', str(config.get('lambda_y', 1.0)),
        '--lambda_c', str(config.get('lambda_c', 1.0)),
        '--lambda_rgb', str(config.get('lambda_rgb', 0.0)),
        '--device', args.device, '--outf', str(Path(args.output_root).resolve()),
        '--run_name', f"{config['sequence']}_{config.get('config_name', 'evaluation')}",
        '--save_predictions', '--results_csv',
        str(Path(args.output_root).resolve() / 'evaluation.csv'),
    ]
    if args.allow_resize:
        command.append('--allow_resize')
    if not args.skip_vmaf:
        command.append('--compute_vmaf')
        if args.vmaf_model_path:
            command.extend(['--vmaf_model_path', args.vmaf_model_path])
    return command


def main():
    args = parse_args()
    command = build_command(args, read_json(args.config))
    print(subprocess.list2cmdline(command))
    return_code = subprocess.run(command, cwd=REPO_ROOT).returncode
    if return_code:
        raise SystemExit(return_code)
    output = Path(args.output_root).resolve()
    metrics_path = output / 'eval_metrics.json'
    metrics = read_json(metrics_path)
    references = sorted((output / 'references').glob('*.png'))
    predictions = sorted((output / 'predictions').glob('*.png'))
    if args.compute_fid:
        metric_device = ('cuda' if args.device == 'auto' and __import__('torch').cuda.is_available()
                         else 'cpu' if args.device == 'auto' else args.device)
        metrics['fid'] = compute_pooled_fid(references, predictions, metric_device)
    if args.compute_kid:
        metric_device = ('cuda' if args.device == 'auto' and __import__('torch').cuda.is_available()
                         else 'cpu' if args.device == 'auto' else args.device)
        metrics.update(compute_pooled_kid(references, predictions, metric_device))
    if args.vmaf_neg_model_path:
        report = compute_vmaf(
            output / 'references/frame_%06d.png',
            output / 'predictions/frame_%06d.png',
            output / 'vmaf_neg.json', output / 'vmaf_neg_command.txt',
            model_path=args.vmaf_neg_model_path)
        metrics['vmaf_neg'] = report['vmaf_score']
    atomic_write_json(metrics_path, metrics)


if __name__ == '__main__':
    main()
