"""Shared validation rules for the anonymous supplementary archive."""

from __future__ import annotations

import re
from pathlib import Path


RELEASE_PATTERNS = (
    'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'requirements.txt',
    'environment.yml', '.gitignore', 'train_chroma_nerv.py', 'model_nerv.py',
    'model_chroma_nerv.py', 'nerv_generalization.py', 'persistence.py', 'utils.py',
    'release_tools.py', 'configs/supplementary/*.json', 'metrics/*.py',
    'scripts/run_nerv_generalization.py', 'scripts/run_bunny_experiments.py',
    'scripts/evaluate_checkpoint.py', 'scripts/aggregate_supplementary_results.py',
    'scripts/validate_release.py', 'scripts/build_supplementary_release.py',
    'tests/*.py', 'tests/fixtures/tiny_video/Beauty/*.png',
    'results/supplementary/*.csv', 'results/supplementary/*.json',
    'results/supplementary/latex/*.tex',
)

FORBIDDEN_PATTERNS = {
    'repository owner': re.compile('Hannan' + 'Akhtar', re.IGNORECASE),
    'student identifier': re.compile('b001' + '01092', re.IGNORECASE),
    'Windows user path': re.compile(r'[A-Za-z]:\\Users\\'),
    'Linux user path': re.compile(r'/home/[^/\s]+/'),
    'personal cloud path': re.compile(r'(?:OneDrive\s*-|/content/drive/MyDrive/)', re.IGNORECASE),
    'macOS user path': re.compile(r'/Users/[^/\s]+/'),
    'private IPv4 address': re.compile(r'(?<!\d)10(?:\.\d{1,3}){3}(?!\d)'),
    'email address': re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b'),
    'credential assignment': re.compile(
        r'(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*["\'][^"\']{8,}'),
}


def release_files(root: Path) -> list[Path]:
    """Resolve the explicit release allowlist without duplicates."""
    files: set[Path] = set()
    for pattern in RELEASE_PATTERNS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def scan_anonymity(root: Path, files: list[Path] | None = None) -> list[str]:
    """Return forbidden-pattern findings for text files in the allowlist."""
    findings: list[str] = []
    for path in files or release_files(root):
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.zip'}:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            findings.append(f'{path.relative_to(root)}: unexpected binary file')
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                findings.append(f'{path.relative_to(root)}: {label}')
    return findings
