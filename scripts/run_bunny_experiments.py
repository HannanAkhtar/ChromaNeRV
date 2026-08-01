#!/usr/bin/env python
"""Run the six-configuration Bunny supplementary grid."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_nerv_generalization import main  # noqa: E402


if __name__ == '__main__':
    if '--config' not in sys.argv:
        sys.argv[1:1] = [
            '--config',
            str(REPO_ROOT / 'configs' / 'supplementary' / 'nerv_bunny.json'),
        ]
    main()
