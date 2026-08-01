#!/usr/bin/env python
"""Build the anonymous, allowlisted NeRV supplementary ZIP."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import sha256_file  # noqa: E402
from release_tools import release_files, scan_anonymity  # noqa: E402
from scripts.validate_release import check_structure  # noqa: E402

ARCHIVE_NAME = 'ChromaNeRV_NeRV_Supplementary.zip'
MAX_FILE_BYTES = 5 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry_run', action='store_true', help='Validate and list files without writing a ZIP')
    return parser.parse_args()


def run_tests() -> None:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT)
    if result.returncode:
        raise RuntimeError('Unit tests failed; release archive was not built')


def archive_manifest(files: list[Path]) -> dict:
    return {
        'archive': ARCHIVE_NAME,
        'scope': 'NeRV generalization supplementary experiments only',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'file_count': len(files) + 1,
        'files': [
            {
                'path': path.relative_to(ROOT).as_posix(),
                'bytes': path.stat().st_size,
                'sha256': sha256_file(path),
            }
            for path in files
        ],
        'exclusions': [
            '.git and remotes', 'datasets', 'checkpoints', 'predictions and references',
            'output directories', 'virtual environments and caches', 'notebooks',
            'archived implementation specifications', 'private logs',
            'main-paper pipelines and results',
        ],
    }


def main() -> None:
    args = parse_args()
    check_structure()
    run_tests()
    files = release_files(ROOT)
    oversized = [path for path in files if path.stat().st_size > MAX_FILE_BYTES]
    if oversized:
        raise RuntimeError('Unexpected file larger than 5 MiB: ' + ', '.join(map(str, oversized)))
    findings = scan_anonymity(ROOT, files)
    if findings:
        raise RuntimeError('Anonymization scan failed:\n' + '\n'.join(findings))
    print('Included files:')
    for path in files:
        print('  ' + path.relative_to(ROOT).as_posix())
    if args.dry_run:
        print(f'Dry run complete: {len(files)} files would be included')
        return

    dist = ROOT / 'dist'
    dist.mkdir(parents=True, exist_ok=True)
    destination = dist / ARCHIVE_NAME
    manifest = archive_manifest(files)
    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = Path(temporary) / 'release_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.write(manifest_path, 'release_manifest.json')
    checksum = sha256_file(destination)
    destination.with_suffix(destination.suffix + '.sha256').write_text(
        f'{checksum}  {destination.name}\n', encoding='ascii')
    print(f'ZIP: {destination}')
    print(f'SHA-256: {checksum}')
    print(f'Size: {destination.stat().st_size} bytes')


if __name__ == '__main__':
    main()
