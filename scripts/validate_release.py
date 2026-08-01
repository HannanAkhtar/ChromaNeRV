#!/usr/bin/env python
"""Validate dependencies, datasets, and anonymous release structure."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nerv_generalization import UVG_SEQUENCES, discover_sequence_frames  # noqa: E402
from release_tools import release_files, scan_anonymity  # noqa: E402

REQUIRED = (
    'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'requirements.txt',
    'environment.yml', 'configs/supplementary/nerv_bunny.json',
    'configs/supplementary/nerv_uvg7.json', 'configs/supplementary/smoke_test.json',
    'scripts/run_nerv_generalization.py', 'scripts/run_bunny_experiments.py',
    'scripts/evaluate_checkpoint.py', 'scripts/aggregate_supplementary_results.py',
    'scripts/build_supplementary_release.py',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check_environment', action='store_true')
    parser.add_argument('--check_data', action='store_true')
    parser.add_argument('--bunny_root')
    parser.add_argument('--uvg_root')
    parser.add_argument('--all', action='store_true')
    return parser.parse_args()


def module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, '__version__', 'available'))
    except ImportError:
        return 'not installed'


def check_environment() -> None:
    print(f'Python: {sys.version.split()[0]}')
    try:
        import torch
        print(f'PyTorch: {torch.__version__}')
        print(f'CUDA available: {torch.cuda.is_available()}')
        print(f'CUDA runtime: {torch.version.cuda or "none"}')
        print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}')
    except ImportError:
        print('PyTorch: not installed')
    print(f'torchvision: {module_version("torchvision")}')
    ffmpeg = shutil.which('ffmpeg')
    print(f'FFmpeg: {ffmpeg or "not found"}')
    libvmaf = False
    if ffmpeg:
        result = subprocess.run([ffmpeg, '-hide_banner', '-filters'], capture_output=True, text=True)
        libvmaf = 'libvmaf' in (result.stdout + result.stderr)
    print(f'libvmaf: {libvmaf}')
    for label, module in (('LPIPS', 'lpips'), ('DISTS', 'DISTS_pytorch')):
        print(f'{label}: {"available" if importlib.util.find_spec(module) else "not installed"}')
    fid = importlib.util.find_spec('torchmetrics') or importlib.util.find_spec('torch_fidelity')
    print(f'FID/KID: {"available" if fid else "not installed"}')


def describe_sequence(root: Path, sequence: str, target: tuple[int, int]) -> None:
    frames = discover_sequence_frames(root, sequence, 132)
    resolutions = set()
    for frame in frames:
        with Image.open(frame) as image:
            resolutions.add((image.height, image.width))
    if len(resolutions) != 1:
        raise RuntimeError(f'{sequence} has inconsistent selected-frame resolutions: {resolutions}')
    resolution = resolutions.pop()
    crop = 'none' if resolution == target else f'deterministic center crop to {target}'
    print(f'{sequence}: {len(frames)} selected; first={frames[0].name}; last={frames[-1].name}; '
          f'source={resolution}; target={target}; preprocessing={crop}')


def check_data(bunny_root: str | None, uvg_root: str | None) -> None:
    if not bunny_root or not uvg_root:
        raise ValueError('--check_data requires --bunny_root and --uvg_root')
    describe_sequence(Path(bunny_root), 'Bunny', (720, 1280))
    for sequence in UVG_SEQUENCES:
        describe_sequence(Path(uvg_root), sequence, (960, 1920))


def check_structure() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError('Missing required release files: ' + ', '.join(missing))
    for path in (ROOT / 'configs/supplementary').glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'))
    findings = scan_anonymity(ROOT, release_files(ROOT))
    if findings:
        raise RuntimeError('Anonymization scan failed:\n' + '\n'.join(findings))
    for module in ('model_nerv', 'model_chroma_nerv', 'nerv_generalization',
                   'metrics.main_paper_metrics'):
        __import__(module)
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for command in ('validate_release.py', 'run_nerv_generalization.py',
                    'run_bunny_experiments.py', 'evaluate_checkpoint.py',
                    'aggregate_supplementary_results.py', 'build_supplementary_release.py'):
        if command not in readme:
            raise RuntimeError(f'README does not reference {command}')
    print(f'Release structure: valid ({len(release_files(ROOT))} allowlisted files)')


def main() -> None:
    args = parse_args()
    if not any((args.check_environment, args.check_data, args.all)):
        raise SystemExit('Select --check_environment, --check_data, or --all')
    if args.check_environment or args.all:
        check_environment()
    if args.check_data:
        check_data(args.bunny_root, args.uvg_root)
    if args.all:
        check_structure()


if __name__ == '__main__':
    main()
