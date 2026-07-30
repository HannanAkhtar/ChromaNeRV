#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metrics.main_paper_metrics import compute_pooled_fid  # noqa: E402
from nerv_generalization import (  # noqa: E402
    DEFAULT_CONFIGS,
    PAPER_CONFIGS,
    UVG_SEQUENCES,
    build_job_config,
    configs_match,
    environment_info,
    parse_csv_names,
    run_is_complete,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the NeRV UVG7 generalization grid')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_root', default='output/nerv_generalization')
    parser.add_argument('--sequences', default=','.join(UVG_SEQUENCES))
    parser.add_argument('--configs', default=','.join(DEFAULT_CONFIGS))
    parser.add_argument('--max_frames', type=int, default=132)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--keep_predictions', action='store_true')
    parser.add_argument('--skip_vmaf', action='store_true')
    parser.add_argument('--skip_fid', action='store_true')
    parser.add_argument('--rate_eval', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint_policy', choices=['final', 'best'], default='final')
    parser.add_argument('--cuda_deterministic', action='store_true')
    return parser.parse_args()


def build_command(args, job, run_dir, config_path):
    architecture = job['architecture']
    command = [
        sys.executable, str(REPO_ROOT / 'train_chroma_nerv.py'),
        '--experiment', job['experiment'],
        '--color_space', job['color_space'],
        '--data_root', str(Path(args.data_root).resolve()),
        '--sequence', job['sequence'],
        '--max_frames', str(job['max_frames']),
        '--target_height', str(architecture['target_height']),
        '--target_width', str(architecture['target_width']),
        '--embed', architecture['embed'],
        '--stem_dim_num', architecture['stem_dim_num'],
        '--fc_hw_dim', architecture['fc_hw_dim'],
        '--expansion', str(architecture['expansion']),
        '--reduction', str(architecture['reduction']),
        '--lower-width', str(architecture['lower_width']),
        '--num-blocks', str(architecture['num_blocks']),
        '--norm', architecture['norm'],
        '--act', architecture['act'],
        '--conv_type', architecture['conv_type'],
        '--strides', *map(str, architecture['strides']),
        '--single_res',
        '--y_branch_width', str(job['branch_width'] or architecture['lower_width']),
        '--rgb_branch_width', str(job['branch_width'] or architecture['lower_width']),
        '--lambda_y', str(job['lambda_y']),
        '--lambda_c', str(job['lambda_c']),
        '--lambda_rgb', str(job['lambda_rgb']),
        '--chroma_downsample', 'area',
        '--chroma_upsample', 'bilinear',
        '--epochs', str(job['epochs']),
        '--batchSize', '1',
        '--workers', '0' if args.smoke_test else '4',
        '--lr', '0.0005',
        '--beta', '0.5',
        '--warmup', '0.2',
        '--lr_type', 'cosine',
        '--loss_type', 'L2',
        '--manualSeed', str(job['seed']),
        '--frame_gap', '1',
        '--test_gap', '1',
        '--device', args.device,
        '--checkpoint_policy', args.checkpoint_policy,
        '--outf', str(run_dir),
        '--run_name', f"{job['sequence']}_{job['config_name']}",
        '--job_config_json', str(config_path),
        '--results_csv', str(Path(args.output_root) / 'all_evaluations.csv'),
        '--fps_warmup', '1' if args.smoke_test else '20',
        '--fps_repeats', '1' if args.smoke_test else '100',
        '--eval_freq', str(job['epochs']),
        '--save_predictions',
    ]
    if args.smoke_test:
        command.extend(['--allow_resize', '--debug'])
    if not args.skip_vmaf:
        command.append('--compute_vmaf')
    if args.resume:
        command.append('--resume')
    if args.rate_eval:
        command.append('--rate_eval')
    if args.cuda_deterministic:
        command.append('--cuda_deterministic')
    if args.eval_only:
        checkpoint_name = (
            'model_final.pth'
            if args.checkpoint_policy == 'final' else 'model_best.pth'
        )
        command.extend(['--eval_only', '--weight', str(run_dir / checkpoint_name)])
    return command


def run_logged(command, log_path):
    with Path(log_path).open('a', encoding='utf-8') as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            log.write(line)
        return process.wait()


def main():
    args = parse_args()
    sequences = parse_csv_names(args.sequences, UVG_SEQUENCES, 'sequence')
    configs = parse_csv_names(args.configs, DEFAULT_CONFIGS, 'configuration')
    if args.smoke_test and args.epochs > 1:
        raise ValueError('--smoke_test permits at most one epoch')
    jobs = [
        build_job_config(sequence, config, args.max_frames, args.epochs, args.seed, args.smoke_test)
        for sequence in sequences for config in configs
    ]
    for job in jobs:
        job['checkpoint_policy'] = args.checkpoint_policy
    output_root = Path(args.output_root)
    manifest_dir = output_root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / 'experiment_grid.json', jobs)
    write_json(manifest_dir / 'environment.json', environment_info(REPO_ROOT))
    print(json.dumps(jobs, indent=2))
    print(f'Planned jobs: {len(jobs)}')
    if args.dry_run:
        return

    for job in jobs:
        run_dir = output_root / job['sequence'] / job['config_name']
        config_path = run_dir / 'config.json'
        if (
                run_is_complete(run_dir, job, args.checkpoint_policy)
                and not args.force
                and not args.eval_only):
            print(f"SKIP complete: {job['sequence']}/{job['config_name']}")
            continue
        if args.eval_only:
            checkpoint_name = (
                'model_final.pth'
                if args.checkpoint_policy == 'final' else 'model_best.pth'
            )
            if not configs_match(config_path, job):
                raise RuntimeError(f'Evaluation config does not match {config_path}')
            if not (run_dir / checkpoint_name).is_file():
                raise RuntimeError(f'Evaluation checkpoint not found: {run_dir / checkpoint_name}')
            command = build_command(args, job, run_dir, config_path)
            print(subprocess.list2cmdline(command))
            if run_logged(command, run_dir / 'eval_log.txt'):
                raise RuntimeError(
                    f"Evaluation failed: {job['sequence']}/{job['config_name']}")
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            if args.force:
                shutil.rmtree(run_dir)
            elif not args.resume:
                raise RuntimeError(
                    f'Incomplete or mismatched run exists at {run_dir}. '
                    'Use --resume to continue it or --force to replace it.')
            elif config_path.exists() and not configs_match(config_path, job):
                raise RuntimeError(
                    f'Cannot resume {run_dir}: its config.json does not match the current job.')
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_path, job)
        command = build_command(args, job, run_dir, config_path)
        print(subprocess.list2cmdline(command))
        return_code = run_logged(command, run_dir / 'train_log.txt')
        if return_code:
            raise RuntimeError(f"Job failed: {job['sequence']}/{job['config_name']}")

    if not args.skip_fid:
        rows = []
        for config in configs:
            reference_paths = []
            prediction_paths = []
            for sequence in sequences:
                run_dir = output_root / sequence / config
                reference_paths.extend(sorted((run_dir / 'references').glob('*.png')))
                prediction_paths.extend(sorted((run_dir / 'predictions').glob('*.png')))
            expected = len(sequences) * args.max_frames
            if len(reference_paths) != expected or len(prediction_paths) != expected:
                raise RuntimeError(
                    f'Pooled FID for {config} expected {expected} image pairs, found '
                    f'{len(reference_paths)} references and {len(prediction_paths)} predictions.')
            fid_device = (
                args.device if args.device != 'auto'
                else ('cuda' if torch.cuda.is_available() else 'cpu')
            )
            score = compute_pooled_fid(
                reference_paths, prediction_paths, device=fid_device)
            rows.append({'config_name': config, 'fid': score, 'pooled_frame_count': expected})
        write_json(output_root / 'manifests' / 'pooled_fid.json', rows)

    if not args.keep_predictions and not args.skip_fid:
        for job in jobs:
            run_dir = output_root / job['sequence'] / job['config_name']
            for name in ('predictions', 'references'):
                path = run_dir / name
                if path.exists():
                    shutil.rmtree(path)


if __name__ == '__main__':
    main()
