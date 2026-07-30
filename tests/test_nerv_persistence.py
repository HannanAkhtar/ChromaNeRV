import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from persistence import (
    atomic_copy_file,
    atomic_write_json,
    persist_resume_checkpoint,
    persist_run_artifacts,
    persistent_resume_is_valid,
    persistent_run_is_complete,
    restore_run_artifacts,
    retry_operation,
    sha256_file,
    stable_config_hash,
    validate_persistent_run,
)
from scripts.aggregate_nerv_generalization import load_result
from scripts.run_nerv_generalization import (
    local_run_is_complete,
    persist_with_retries,
)


def sample_config():
    return {
        'sequence': 'Beauty',
        'config_name': 'chroma_w8',
        'architecture': {'strides': [2, 2, 2], 'target_height': 16, 'target_width': 32},
        'branch_width': 8,
        'lambda_y': 1.0,
        'lambda_c': 1.0,
        'lambda_rgb': 0.1,
        'epochs': 1,
        'seed': 1,
        'checkpoint_policy': 'final',
        'selected_frame_names': ['Beauty/frame1.png', 'Beauty/frame2.png'],
        'training': {'optimizer': 'Adam', 'learning_rate': 5e-4},
        'metrics': {'vmaf_enabled': False, 'fid_enabled': False},
        'local_run_dir': 'machine-a/local',
        'persistent_run_dir': 'machine-a/onedrive',
        'persistence_enabled': True,
    }


def create_local_run(path, config=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config = config or sample_config()
    atomic_write_json(path / 'config.json', config)
    atomic_write_json(path / 'environment.json', {'git_commit': 'abc'})
    atomic_write_json(path / 'eval_metrics.json', {'rgb_psnr': 30.0})
    (path / 'command.txt').write_text('python train.py\n', encoding='utf-8')
    (path / 'selected_frames.txt').write_text(
        'Beauty/frame1.png\nBeauty/frame2.png\n', encoding='utf-8')
    (path / 'model_final.pth').write_bytes(b'checkpoint-final')
    (path / 'model_latest.pth').write_bytes(b'checkpoint-latest')
    (path / 'per_frame_metrics.csv').write_text(
        'frame_index,rgb_psnr\n0,30\n', encoding='utf-8')
    (path / 'train_log.txt').write_text('trained\n', encoding='utf-8')
    return path


class PersistenceTests(unittest.TestCase):
    def test_atomic_json_write_and_temp_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'value.json'
            atomic_write_json(path, {'value': 3})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'value': 3})
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_atomic_checkpoint_copy_preserves_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pth'
            destination = Path(directory) / 'nested' / 'copy.pth'
            source.write_bytes(b'\x00checkpoint\xff')
            atomic_copy_file(source, destination)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(sha256_file(source), sha256_file(destination))
            self.assertFalse(Path(str(destination) + '.tmp').exists())

    def test_directory_without_completion_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(persistent_run_is_complete(directory, sample_config()))

    def test_early_completion_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory) / 'completion.json', {'status': 'complete'})
            valid, reason = validate_persistent_run(directory, sample_config())
            self.assertFalse(valid)
            self.assertIn('config', reason)

    def test_valid_persistent_run_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            persistent = Path(directory) / 'persistent'
            persist_run_artifacts(local, persistent, sample_config(), git_commit='abc')
            self.assertTrue(persistent_run_is_complete(persistent, sample_config()))
            self.assertEqual(
                json.loads((persistent / 'completion.json').read_text())['status'],
                'complete',
            )

    def test_machine_paths_do_not_change_scientific_hash(self):
        first = sample_config()
        second = dict(first, local_run_dir='machine-b/local', persistent_run_dir='other')
        self.assertEqual(stable_config_hash(first), stable_config_hash(second))

    def test_mismatched_configuration_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            persistent = Path(directory) / 'persistent'
            persist_run_artifacts(local, persistent, sample_config())
            completion = json.loads((persistent / 'completion.json').read_text())
            completion['config_hash'] = '0' * 64
            atomic_write_json(persistent / 'completion.json', completion)
            self.assertFalse(persistent_run_is_complete(persistent, sample_config()))

    def test_mismatched_hash_inside_persistent_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            persistent = Path(directory) / 'persistent'
            persist_run_artifacts(local, persistent, sample_config())
            config = json.loads((persistent / 'config.json').read_text())
            config['config_hash'] = 'f' * 64
            atomic_write_json(persistent / 'config.json', config)
            valid, reason = validate_persistent_run(persistent, sample_config())
            self.assertFalse(valid)
            self.assertIn('config.json scientific hash mismatch', reason)

    def test_corrupted_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            persistent = Path(directory) / 'persistent'
            persist_run_artifacts(local, persistent, sample_config())
            with (persistent / 'model_final.pth').open('ab') as handle:
                handle.write(b'corrupt')
            self.assertFalse(persistent_run_is_complete(persistent, sample_config()))

    def test_corrupted_eval_metrics_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            persistent = Path(directory) / 'persistent'
            persist_run_artifacts(local, persistent, sample_config())
            (persistent / 'eval_metrics.json').write_text('{}', encoding='utf-8')
            self.assertFalse(persistent_run_is_complete(persistent, sample_config()))

    def test_local_complete_can_be_persisted_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            persistent = Path(directory) / 'persistent'
            self.assertTrue(local_run_is_complete(local, sample_config(), 'final'))
            persist_run_artifacts(local, persistent, sample_config())
            self.assertTrue(persistent_run_is_complete(persistent, sample_config()))

    def test_persistent_complete_supports_missing_local_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'source')
            persistent = Path(directory) / 'persistent'
            persist_run_artifacts(local, persistent, sample_config())
            missing_local = Path(directory) / 'lost'
            self.assertFalse(local_run_is_complete(missing_local, sample_config(), 'final'))
            self.assertTrue(persistent_run_is_complete(persistent, sample_config()))

    def test_resumable_checkpoint_restores_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'source')
            persistent = Path(directory) / 'persistent'
            persistent.mkdir()
            config = sample_config()
            atomic_write_json(
                persistent / 'config.json',
                {**config, 'config_hash': stable_config_hash(config)},
            )
            persist_resume_checkpoint(local, persistent, config, epoch=10)
            self.assertTrue(persistent_resume_is_valid(persistent, config))
            restored = Path(directory) / 'restored'
            restore_run_artifacts(persistent, restored, config, resume_only=True)
            self.assertEqual(
                (restored / 'model_latest.pth').read_bytes(),
                (local / 'model_latest.pth').read_bytes(),
            )

    def test_incomplete_persistent_state_does_not_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            persistent = Path(directory)
            atomic_write_json(persistent / 'config.json', sample_config())
            (persistent / 'model_final.pth').write_bytes(b'incomplete')
            self.assertFalse(persistent_run_is_complete(persistent, sample_config()))

    def test_predictions_are_excluded_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            (local / 'predictions').mkdir()
            (local / 'references').mkdir()
            (local / 'predictions' / 'frame.png').write_bytes(b'png')
            (local / 'references' / 'frame.png').write_bytes(b'png')
            persistent = Path(directory) / 'persistent'
            persist_run_artifacts(local, persistent, sample_config())
            self.assertFalse((persistent / 'predictions').exists())
            self.assertFalse((persistent / 'references').exists())

    def test_persist_predictions_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            (local / 'predictions').mkdir()
            (local / 'references').mkdir()
            (local / 'predictions' / 'frame.png').write_bytes(b'prediction')
            (local / 'references' / 'frame.png').write_bytes(b'reference')
            persistent = Path(directory) / 'persistent'
            persist_run_artifacts(
                local, persistent, sample_config(), persist_predictions=True)
            self.assertTrue((persistent / 'predictions' / 'frame.png').is_file())
            self.assertTrue((persistent / 'references' / 'frame.png').is_file())

    def test_pooled_fid_json_can_be_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifests' / 'pooled_fid.json'
            atomic_write_json(path, [{'config_name': 'chroma_w8', 'fid': 1.2}])
            self.assertEqual(json.loads(path.read_text())[0]['fid'], 1.2)
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_aggregation_falls_back_to_valid_persistent_result(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'source')
            persistent_root = Path(directory) / 'persistent-root'
            persistent = persistent_root / 'runs' / 'Beauty' / 'chroma_w8'
            persist_run_artifacts(local, persistent, sample_config())
            row = load_result(
                Path(directory) / 'missing-local',
                persistent_root,
                'Beauty',
                'chroma_w8',
            )
            self.assertEqual(row['result_source'], 'persistent')
            self.assertEqual(row['rgb_psnr'], 30.0)

    def test_aggregation_rejects_invalid_persistent_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'persistent'
            run = root / 'runs' / 'Beauty' / 'chroma_w8'
            run.mkdir(parents=True)
            atomic_write_json(run / 'completion.json', {'status': 'complete'})
            with self.assertRaisesRegex(RuntimeError, 'Invalid persistent result'):
                load_result(Path(directory) / 'local', root, 'Beauty', 'chroma_w8')

    def test_retry_operation_retries_then_succeeds(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise OSError('OneDrive unavailable')
            return 'ok'

        self.assertEqual(retry_operation(flaky, retries=3, retry_delay=0), 'ok')
        self.assertEqual(len(attempts), 3)

    def test_failed_persistence_writes_pending_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            local = create_local_run(Path(directory) / 'local')
            args = SimpleNamespace(
                checkpoint_policy='final',
                persist_predictions=False,
                skip_vmaf=True,
                persistence_retries=2,
                persistence_retry_delay=0,
                continue_on_persistence_failure=True,
            )
            with mock.patch(
                    'scripts.run_nerv_generalization.persist_run_artifacts',
                    side_effect=OSError('offline')):
                result = persist_with_retries(
                    args,
                    sample_config(),
                    local,
                    Path(directory) / 'persistent',
                )
            self.assertIsNone(result)
            self.assertTrue((local / 'persistence_pending.json').is_file())


if __name__ == '__main__':
    unittest.main()
