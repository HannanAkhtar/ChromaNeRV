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
    UVG_SEQUENCES,
    build_job_config,
    discover_sequence_frames,
    environment_info,
    git_commit,
    inspect_frame_resolution,
    parse_csv_names,
    read_json,
    resolve_spatial_preprocessing,
    write_json,
)
from persistence import (  # noqa: E402
    atomic_write_json,
    configs_scientifically_match,
    persist_resume_checkpoint,
    persist_run_artifacts,
    persistence_config,
    persistent_resume_is_valid,
    restore_run_artifacts,
    retry_operation,
    stable_config_hash,
    validate_persistent_run,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the NeRV UVG7 generalization grid')
    parser.add_argument('--config', default=None, help='Supplementary experiment JSON config')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_root', default='output/nerv_generalization')
    parser.add_argument('--persistent_root', default=None)
    parser.add_argument('--sequences', default=None)
    parser.add_argument('--configs', default=None)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--allow_resize', action='store_true')
    parser.add_argument('--keep_predictions', action='store_true')
    parser.add_argument('--skip_vmaf', action='store_true')
    parser.add_argument('--skip_fid', action='store_true')
    parser.add_argument('--rate_eval', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint_policy', choices=['final', 'best'], default=None)
    parser.add_argument('--cuda_deterministic', action='store_true')
    parser.add_argument('--persist_predictions', action='store_true')
    parser.add_argument('--persist_checkpoint_interval', type=int, default=10)
    parser.add_argument('--persistence_retries', type=int, default=3)
    parser.add_argument('--persistence_retry_delay', type=float, default=10)
    parser.add_argument('--continue_on_persistence_failure', action='store_true')
    return parser.parse_args()


def build_command(args, job, run_dir, config_path, evaluation_only=False):
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
    if args.smoke_test or args.allow_resize:
        command.append('--allow_resize')
    if args.smoke_test:
        command.append('--debug')
    if not args.skip_vmaf:
        command.append('--compute_vmaf')
    if args.resume:
        command.append('--resume')
    if args.rate_eval:
        command.append('--rate_eval')
    if args.cuda_deterministic:
        command.append('--cuda_deterministic')
    if args.persistent_root:
        command.extend([
            '--persistent_run_dir', job['persistent_run_dir'],
            '--persist_checkpoint_interval', str(args.persist_checkpoint_interval),
            '--persistence_retries', str(args.persistence_retries),
            '--persistence_retry_delay', str(args.persistence_retry_delay),
        ])
    if args.eval_only or evaluation_only:
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


def local_run_is_complete(run_dir, expected_config, checkpoint_policy):
    run_dir = Path(run_dir)
    checkpoint = (
        'model_final.pth' if checkpoint_policy == 'final' else 'model_best.pth')
    try:
        local_config = read_json(run_dir / 'config.json')
    except (OSError, json.JSONDecodeError):
        return False
    return (
        configs_scientifically_match(local_config, expected_config)
        and (run_dir / checkpoint).is_file()
        and (run_dir / 'eval_metrics.json').is_file()
        and (run_dir / 'per_frame_metrics.csv').is_file()
        and (run_dir / 'train_log.txt').is_file()
    )


def persist_with_retries(args, job, run_dir, persistent_run_dir):
    try:
        completion = retry_operation(
            lambda: persist_run_artifacts(
                run_dir,
                persistent_run_dir,
                job,
                checkpoint_policy=args.checkpoint_policy,
                persist_predictions=args.persist_predictions,
                require_vmaf=not args.skip_vmaf,
                git_commit=git_commit(REPO_ROOT),
            ),
            retries=args.persistence_retries,
            retry_delay=args.persistence_retry_delay,
            on_retry=lambda attempt, exc: print(
                f'Persistence retry {attempt}/{args.persistence_retries}: {exc}'),
        )
        pending_path = Path(run_dir) / 'persistence_pending.json'
        if pending_path.exists():
            pending_path.unlink()
        return completion
    except Exception as exc:
        atomic_write_json(Path(run_dir) / 'persistence_pending.json', {
            'status': 'pending',
            'stage': 'final_artifacts',
            'error': str(exc),
            'persistent_run_dir': str(persistent_run_dir),
        })
        message = (
            f'Persistence failed for {job["sequence"]}/{job["config_name"]}. '
            f'Local artifacts remain intact at {run_dir}. Error: {exc}')
        if args.continue_on_persistence_failure:
            print('WARNING: ' + message)
            return None
        raise RuntimeError(message) from exc


def regenerate_fid_images(args, job, run_dir, config_path):
    expected = job['max_frames']
    predictions = list((run_dir / 'predictions').glob('*.png'))
    references = list((run_dir / 'references').glob('*.png'))
    if len(predictions) == expected and len(references) == expected:
        return
    print(
        'Persistent run is complete but pooled-FID images are absent. '
        'Regenerating predictions from the persisted final checkpoint.')
    command = build_command(
        args, job, run_dir, config_path, evaluation_only=True)
    print(subprocess.list2cmdline(command))
    if run_logged(command, run_dir / 'eval_log.txt'):
        raise RuntimeError(
            f'FID image regeneration failed: {job["sequence"]}/{job["config_name"]}')


def main():
    args = parse_args()
    release_config = read_json(args.config) if args.config else {}
    configured_sequences = release_config.get('sequences', list(UVG_SEQUENCES))
    configured_configs = release_config.get('configs', list(DEFAULT_CONFIGS))
    args.sequences = args.sequences or ','.join(configured_sequences)
    args.configs = args.configs or ','.join(configured_configs)
    args.max_frames = args.max_frames or int(release_config.get('max_frames', 132))
    args.epochs = args.epochs or int(release_config.get('training', {}).get('epochs', 300))
    args.seed = args.seed if args.seed is not None else int(
        release_config.get('training', {}).get('seed', 1))
    args.checkpoint_policy = (
        args.checkpoint_policy or release_config.get('checkpoint_policy', 'final'))
    args.data_root = str(Path(args.data_root).resolve())
    args.output_root = str(Path(args.output_root).resolve())
    if args.persistent_root:
        args.persistent_root = str(Path(args.persistent_root).resolve())
        retry_operation(
            lambda: Path(args.persistent_root).mkdir(parents=True, exist_ok=True),
            retries=args.persistence_retries,
            retry_delay=args.persistence_retry_delay,
        )
    else:
        print(
            'WARNING: --persistent_root was not supplied; automatic persistence is disabled.',
            file=sys.stderr,
        )
    sequences = parse_csv_names(
        args.sequences, ('Bunny',) + UVG_SEQUENCES, 'sequence')
    configs = parse_csv_names(args.configs, DEFAULT_CONFIGS, 'configuration')
    if args.smoke_test and args.epochs > 1:
        raise ValueError('--smoke_test permits at most one epoch')
    base_jobs = [
        build_job_config(sequence, config, args.max_frames, args.epochs, args.seed, args.smoke_test)
        for sequence in sequences for config in configs
    ]
    output_root = Path(args.output_root)
    persistent_root = Path(args.persistent_root) if args.persistent_root else None
    jobs = []
    for job in base_jobs:
        job['checkpoint_policy'] = args.checkpoint_policy
        if release_config.get('architecture'):
            job['architecture'] = dict(release_config['architecture'])
            job['architecture']['target_height'] = release_config['target_resolution'][0]
            job['architecture']['target_width'] = release_config['target_resolution'][1]
        if job['experiment'] == 'neural420_asym_y':
            weights = release_config.get('loss_weights', {})
            job['lambda_y'] = float(weights.get('lambda_y', job['lambda_y']))
            job['lambda_c'] = float(weights.get('lambda_c', job['lambda_c']))
            job['lambda_rgb'] = float(weights.get('lambda_rgb', job['lambda_rgb']))
        if release_config.get('training'):
            job['training'].update(release_config['training'])
        selected = discover_sequence_frames(
            args.data_root, job['sequence'], args.max_frames)
        source_resolution = inspect_frame_resolution(selected)
        architecture = job['architecture']
        preprocessing = resolve_spatial_preprocessing(
            source_resolution,
            (architecture['target_height'], architecture['target_width']),
            allow_resize=args.smoke_test or args.allow_resize,
        )
        job['selected_frame_names'] = [
            str(path.relative_to(Path(args.data_root))).replace('\\', '/')
            for path in selected
        ]
        local_run_dir = output_root / job['sequence'] / job['config_name']
        persistent_run_dir = (
            persistent_root / 'runs' / job['sequence'] / job['config_name']
            if persistent_root else None
        )
        job.update({
            **preprocessing,
            'preprocessing': preprocessing,
            'local_run_dir': str(local_run_dir.resolve()),
            'persistent_run_dir': (
                str(persistent_run_dir.resolve()) if persistent_run_dir else None),
            'persistence_enabled': persistent_root is not None,
            'persist_predictions': args.persist_predictions,
            'persist_checkpoint_interval': args.persist_checkpoint_interval,
            'metrics': {
                'vmaf_enabled': not args.skip_vmaf,
                'vmaf_fps': 30,
                'fid_enabled': not args.skip_fid,
                'lpips_network': 'alex',
                'weighted_yuv_psnr': 'mse_first_6_1_1',
            },
        })
        job['config_hash'] = stable_config_hash(job)
        jobs.append(job)
    manifest_dir = output_root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / 'experiment_grid.json', jobs)
    environment = environment_info(REPO_ROOT)
    write_json(manifest_dir / 'environment.json', environment)
    if persistent_root:
        persistent_manifests = persistent_root / 'manifests'
        startup_config = persistence_config(
            output_root,
            persistent_root,
            args.persist_predictions,
            args.persist_checkpoint_interval,
            args.checkpoint_policy,
            git_commit(REPO_ROOT),
        )
        retry_operation(
            lambda: (
                persistent_manifests.mkdir(parents=True, exist_ok=True),
                atomic_write_json(
                    persistent_manifests / 'experiment_grid.json', jobs),
                atomic_write_json(
                    persistent_manifests / 'environment.json', environment),
                atomic_write_json(
                    persistent_manifests / 'persistence_config.json',
                    startup_config),
            ),
            retries=args.persistence_retries,
            retry_delay=args.persistence_retry_delay,
        )
    print(json.dumps(jobs, indent=2))
    print(f'Planned jobs: {len(jobs)}')
    if args.dry_run:
        return

    for job in jobs:
        run_dir = Path(job['local_run_dir'])
        persistent_run_dir = (
            Path(job['persistent_run_dir']) if persistent_root else None)
        config_path = run_dir / 'config.json'
        if config_path.exists():
            local_config = read_json(config_path)
            if not configs_scientifically_match(local_config, job):
                raise RuntimeError(
                    f'Local configuration does not match requested job: {config_path}')
        if persistent_run_dir and (persistent_run_dir / 'config.json').exists():
            persistent_config = read_json(persistent_run_dir / 'config.json')
            if not configs_scientifically_match(persistent_config, job):
                raise RuntimeError(
                    'Persistent configuration does not match requested job: '
                    f'{persistent_run_dir / "config.json"}')
        elif persistent_run_dir:
            persistent_config = dict(job)
            persistent_config['config_hash'] = stable_config_hash(job)
            retry_operation(
                lambda: (
                    persistent_run_dir.mkdir(parents=True, exist_ok=True),
                    atomic_write_json(
                        persistent_run_dir / 'config.json', persistent_config),
                ),
                retries=args.persistence_retries,
                retry_delay=args.persistence_retry_delay,
            )

        local_complete = local_run_is_complete(
            run_dir, job, args.checkpoint_policy)
        persistent_complete = False
        if persistent_run_dir:
            persistent_complete, reason = validate_persistent_run(
                persistent_run_dir, job, require_vmaf=not args.skip_vmaf)
            if (persistent_run_dir / 'completion.json').exists() and not persistent_complete:
                print(f'WARNING: Invalid persistent completion marker: {reason}')

        pending_path = run_dir / 'persistence_pending.json'
        if pending_path.exists() and persistent_run_dir:
            print(f'Retrying pending persistence before training: {pending_path}')
            if local_complete:
                persist_with_retries(
                    args, job, run_dir, persistent_run_dir)
                persistent_complete = True
            elif (run_dir / 'model_latest.pth').is_file():
                retry_operation(
                    lambda: persist_resume_checkpoint(
                        run_dir, persistent_run_dir, job, 0),
                    retries=args.persistence_retries,
                    retry_delay=args.persistence_retry_delay,
                )
                pending_path.unlink()

        if local_complete and persistent_run_dir and not persistent_complete:
            persist_with_retries(args, job, run_dir, persistent_run_dir)
            persistent_complete = True

        if args.eval_only:
            checkpoint_name = (
                'model_final.pth'
                if args.checkpoint_policy == 'final' else 'model_best.pth'
            )
            if not local_complete and persistent_complete:
                restore_run_artifacts(
                    persistent_run_dir, run_dir, job)
                write_json(config_path, job)
            if not (run_dir / checkpoint_name).is_file():
                raise RuntimeError(f'Evaluation checkpoint not found: {run_dir / checkpoint_name}')
            command = build_command(args, job, run_dir, config_path)
            print(subprocess.list2cmdline(command))
            if run_logged(command, run_dir / 'eval_log.txt'):
                raise RuntimeError(
                    f"Evaluation failed: {job['sequence']}/{job['config_name']}")
            if persistent_run_dir:
                persist_with_retries(args, job, run_dir, persistent_run_dir)
            continue

        if (local_complete or persistent_complete) and not args.force:
            if not args.skip_fid:
                if not local_complete:
                    restore_run_artifacts(
                        persistent_run_dir, run_dir, job)
                    write_json(config_path, job)
                regenerate_fid_images(args, job, run_dir, config_path)
            source = 'local and persistent' if local_complete and persistent_complete else (
                'local' if local_complete else 'persistent')
            print(
                f"SKIP complete ({source}): {job['sequence']}/{job['config_name']}")
            continue

        if run_dir.exists() and any(run_dir.iterdir()):
            if args.force:
                shutil.rmtree(run_dir)
            elif not args.resume:
                raise RuntimeError(
                    f'Incomplete or mismatched run exists at {run_dir}. '
                    'Use --resume to continue it or --force to replace it.')
        if (
                persistent_run_dir
                and not (run_dir / 'model_latest.pth').is_file()
                and persistent_resume_is_valid(persistent_run_dir, job)):
            print(f'Restoring resumable checkpoint from {persistent_run_dir}')
            restore_run_artifacts(
                persistent_run_dir, run_dir, job, resume_only=True)
            args.resume = True
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_path, job)
        command = build_command(args, job, run_dir, config_path)
        print(subprocess.list2cmdline(command))
        return_code = run_logged(command, run_dir / 'train_log.txt')
        if return_code:
            raise RuntimeError(f"Job failed: {job['sequence']}/{job['config_name']}")
        if args.smoke_test:
            reload_command = build_command(
                args, job, run_dir, config_path, evaluation_only=True)
            print('Smoke checkpoint reload: ' + subprocess.list2cmdline(reload_command))
            if run_logged(reload_command, run_dir / 'eval_log.txt'):
                raise RuntimeError(
                    f"Smoke reload evaluation failed: {job['sequence']}/{job['config_name']}")
        if persistent_run_dir:
            persist_with_retries(args, job, run_dir, persistent_run_dir)

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
        if persistent_root:
            atomic_write_json(
                persistent_root / 'manifests' / 'pooled_fid.json', rows)

    if not args.keep_predictions and not args.skip_fid:
        for job in jobs:
            run_dir = output_root / job['sequence'] / job['config_name']
            for name in ('predictions', 'references'):
                path = run_dir / name
                if path.exists():
                    shutil.rmtree(path)


if __name__ == '__main__':
    main()
