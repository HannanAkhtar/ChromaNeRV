import getpass
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MACHINE_SPECIFIC_KEYS = {
    'config_hash',
    'created_at',
    'local_output_root',
    'local_run_dir',
    'persistent_root',
    'persistent_run_dir',
    'persistence_enabled',
    'persist_predictions',
    'persist_checkpoint_interval',
}

REQUIRED_RUN_ARTIFACTS = (
    'config.json',
    'command.txt',
    'environment.json',
    'selected_frames.txt',
    'model_final.pth',
    'eval_metrics.json',
    'per_frame_metrics.csv',
    'train_log.txt',
)

OPTIONAL_RUN_ARTIFACTS = (
    'model_best.pth',
    'model_val_best.pth',
    'eval_log.txt',
    'vmaf.json',
    'vmaf_command.txt',
    'rate_eval.json',
)


def _temporary_path(destination):
    destination = Path(destination)
    return destination.with_name(destination.name + '.tmp')


def _replace_temporary(temp_path, destination):
    os.replace(temp_path, destination)


def atomic_write_text(path, text, encoding='utf-8'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        with temp_path.open('w', encoding=encoding, newline='') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_temporary(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


def atomic_write_json(path, value):
    text = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + '\n'
    return atomic_write_text(path, text)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy_file(source, destination, verify_sha256=True):
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(destination)
    try:
        shutil.copy2(source, temp_path)
        if temp_path.stat().st_size != source.stat().st_size:
            raise IOError(f'Copied size mismatch for {source}')
        if verify_sha256 and sha256_file(temp_path) != sha256_file(source):
            raise IOError(f'Copied SHA-256 mismatch for {source}')
        _replace_temporary(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return destination


def atomic_copy_directory_files(source_dir, destination_dir):
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    copied = []
    if not source_dir.is_dir():
        return copied
    for source in sorted(path for path in source_dir.rglob('*') if path.is_file()):
        relative = source.relative_to(source_dir)
        copied.append(atomic_copy_file(source, destination_dir / relative))
    return copied


def _scientific_value(value):
    if isinstance(value, dict):
        return {
            key: _scientific_value(child)
            for key, child in value.items()
            if key not in MACHINE_SPECIFIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_value(child) for child in value]
    return value


def stable_config_hash(config):
    encoded = json.dumps(
        _scientific_value(config),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def configs_scientifically_match(left, right):
    return stable_config_hash(left) == stable_config_hash(right)


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def checkpoint_filename(checkpoint_policy):
    return 'model_final.pth' if checkpoint_policy == 'final' else 'model_best.pth'


def validate_persistent_run(
        persistent_run_dir,
        expected_config=None,
        require_vmaf=None):
    run_dir = Path(persistent_run_dir)
    completion_path = run_dir / 'completion.json'
    if not completion_path.is_file():
        return False, 'completion.json is missing'
    try:
        completion = read_json(completion_path)
        config = read_json(run_dir / 'config.json')
    except (OSError, json.JSONDecodeError) as exc:
        return False, f'cannot read completion/config JSON: {exc}'
    if completion.get('status') != 'complete':
        return False, 'completion status is not complete'
    config_hash = stable_config_hash(config)
    if config.get('config_hash') != config_hash:
        return False, 'config.json scientific hash mismatch'
    if expected_config is not None and not configs_scientifically_match(config, expected_config):
        return False, 'persistent configuration does not match the requested job'
    if completion.get('config_hash') != config_hash:
        return False, 'configuration hash mismatch'
    checkpoint = run_dir / completion.get('checkpoint_filename', '')
    if not checkpoint.is_file():
        return False, 'requested checkpoint is missing'
    if checkpoint.stat().st_size != completion.get('checkpoint_size_bytes'):
        return False, 'checkpoint size mismatch'
    if sha256_file(checkpoint) != completion.get('checkpoint_sha256'):
        return False, 'checkpoint SHA-256 mismatch'
    metrics_path = run_dir / 'eval_metrics.json'
    if not metrics_path.is_file():
        return False, 'eval_metrics.json is missing'
    if sha256_file(metrics_path) != completion.get('eval_metrics_sha256'):
        return False, 'eval_metrics.json SHA-256 mismatch'
    if not (run_dir / 'per_frame_metrics.csv').is_file():
        return False, 'per_frame_metrics.csv is missing'
    for artifact in completion.get('required_artifacts', []):
        if not (run_dir / artifact).is_file():
            return False, f'required artifact is missing: {artifact}'
    vmaf_enabled = (
        require_vmaf
        if require_vmaf is not None
        else bool(config.get('metrics', {}).get('vmaf_enabled'))
    )
    if vmaf_enabled and not (run_dir / 'vmaf.json').is_file():
        return False, 'vmaf.json is required but missing'
    return True, 'complete'


def persistent_run_is_complete(persistent_run_dir, expected_config=None, require_vmaf=None):
    return validate_persistent_run(
        persistent_run_dir, expected_config, require_vmaf)[0]


def persist_resume_checkpoint(local_run_dir, persistent_run_dir, config, epoch):
    local_run_dir = Path(local_run_dir)
    persistent_run_dir = Path(persistent_run_dir)
    checkpoint = local_run_dir / 'model_latest.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    atomic_copy_file(checkpoint, persistent_run_dir / checkpoint.name)
    atomic_write_json(persistent_run_dir / 'resume.json', {
        'status': 'resumable',
        'epoch': int(epoch),
        'config_hash': stable_config_hash(config),
        'checkpoint_filename': checkpoint.name,
        'checkpoint_size_bytes': checkpoint.stat().st_size,
        'checkpoint_sha256': sha256_file(checkpoint),
        'updated_at': datetime.now(timezone.utc).isoformat(),
    })


def persistent_resume_is_valid(persistent_run_dir, expected_config):
    run_dir = Path(persistent_run_dir)
    try:
        resume = read_json(run_dir / 'resume.json')
        config = read_json(run_dir / 'config.json')
    except (OSError, json.JSONDecodeError):
        return False
    checkpoint = run_dir / resume.get('checkpoint_filename', '')
    return (
        resume.get('status') == 'resumable'
        and configs_scientifically_match(config, expected_config)
        and config.get('config_hash') == stable_config_hash(config)
        and resume.get('config_hash') == stable_config_hash(config)
        and checkpoint.is_file()
        and checkpoint.stat().st_size == resume.get('checkpoint_size_bytes')
        and sha256_file(checkpoint) == resume.get('checkpoint_sha256')
    )


def persist_run_artifacts(
        local_run_dir,
        persistent_run_dir,
        config,
        checkpoint_policy='final',
        persist_predictions=False,
        require_vmaf=False,
        git_commit='unavailable'):
    local_run_dir = Path(local_run_dir).resolve()
    persistent_run_dir = Path(persistent_run_dir).resolve()
    persistent_run_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        name for name in REQUIRED_RUN_ARTIFACTS
        if not (local_run_dir / name).is_file()
    ]
    requested_checkpoint = checkpoint_filename(checkpoint_policy)
    if requested_checkpoint not in REQUIRED_RUN_ARTIFACTS:
        missing.extend(
            [requested_checkpoint]
            if not (local_run_dir / requested_checkpoint).is_file() else [])
    if require_vmaf and not (local_run_dir / 'vmaf.json').is_file():
        missing.append('vmaf.json')
    if missing:
        raise RuntimeError(
            f'Cannot persist incomplete run {local_run_dir}; missing: {", ".join(missing)}')

    config_to_persist = dict(config)
    config_to_persist['config_hash'] = stable_config_hash(config)
    atomic_write_json(persistent_run_dir / 'config.json', config_to_persist)
    for name in REQUIRED_RUN_ARTIFACTS:
        if name == 'config.json':
            continue
        atomic_copy_file(local_run_dir / name, persistent_run_dir / name)
    for name in OPTIONAL_RUN_ARTIFACTS:
        source = local_run_dir / name
        if source.is_file():
            atomic_copy_file(source, persistent_run_dir / name)
    if persist_predictions:
        atomic_copy_directory_files(
            local_run_dir / 'predictions', persistent_run_dir / 'predictions')
        atomic_copy_directory_files(
            local_run_dir / 'references', persistent_run_dir / 'references')

    checkpoint = persistent_run_dir / requested_checkpoint
    metrics = persistent_run_dir / 'eval_metrics.json'
    required = list(REQUIRED_RUN_ARTIFACTS)
    if checkpoint_policy == 'best' and 'model_best.pth' not in required:
        required.append('model_best.pth')
    if require_vmaf:
        required.append('vmaf.json')
    completion = {
        'status': 'complete',
        'sequence': config['sequence'],
        'config_name': config['config_name'],
        'git_commit': git_commit,
        'config_hash': stable_config_hash(config),
        'checkpoint_policy': checkpoint_policy,
        'checkpoint_filename': requested_checkpoint,
        'checkpoint_size_bytes': checkpoint.stat().st_size,
        'checkpoint_sha256': sha256_file(checkpoint),
        'eval_metrics_sha256': sha256_file(metrics),
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'required_artifacts': required,
        'local_run_dir': str(local_run_dir),
        'persistent_run_dir': str(persistent_run_dir),
    }
    atomic_write_json(persistent_run_dir / 'completion.json', completion)
    valid, reason = validate_persistent_run(
        persistent_run_dir, config, require_vmaf)
    if not valid:
        raise RuntimeError(f'Persisted run failed validation: {reason}')
    return completion


def restore_run_artifacts(
        persistent_run_dir,
        local_run_dir,
        expected_config,
        resume_only=False):
    persistent_run_dir = Path(persistent_run_dir)
    local_run_dir = Path(local_run_dir)
    local_run_dir.mkdir(parents=True, exist_ok=True)
    if resume_only:
        if not persistent_resume_is_valid(persistent_run_dir, expected_config):
            raise RuntimeError(f'Persistent resume checkpoint is invalid: {persistent_run_dir}')
        names = ('config.json', 'model_latest.pth', 'resume.json')
    else:
        valid, reason = validate_persistent_run(
            persistent_run_dir, expected_config)
        if not valid:
            raise RuntimeError(f'Persistent run is invalid: {reason}')
        names = tuple(REQUIRED_RUN_ARTIFACTS) + OPTIONAL_RUN_ARTIFACTS + ('completion.json',)
    for name in names:
        source = persistent_run_dir / name
        if source.is_file():
            atomic_copy_file(source, local_run_dir / name)
    return local_run_dir


def retry_operation(operation, retries=3, retry_delay=10, on_retry=None):
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts:
                raise
            if on_retry:
                on_retry(attempt, exc)
            time.sleep(retry_delay)


def persistence_config(
        local_output_root,
        persistent_root,
        persist_predictions,
        persist_checkpoint_interval,
        checkpoint_policy,
        git_commit):
    return {
        'local_output_root': str(Path(local_output_root).resolve()),
        'persistent_root': str(Path(persistent_root).resolve()),
        'persistence_enabled': True,
        'persist_predictions': bool(persist_predictions),
        'persist_checkpoint_interval': int(persist_checkpoint_interval),
        'checkpoint_policy': checkpoint_policy,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit,
        'host': socket.gethostname(),
        'user': getpass.getuser(),
        'python_executable': sys.executable,
    }
