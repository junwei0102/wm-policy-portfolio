"""Launcher for policy-bank training via the UNMODIFIED OGBench impls/main.py.

Compute nodes have no internet, but impls hard-codes wandb mode='online'.
This wrapper patches `main.setup_wandb` (the name imported into main's
namespace) to force the mode from $WMPP_WANDB_MODE (default: offline) without
touching the OGBench checkout. Sync offline runs later with `wandb sync`.

Usage: python train_policy.py <all impls/main.py flags>
Requires: OGBENCH_IMPLS env var pointing at the impls/ directory.
"""

import os
import sys

IMPLS = os.environ.get('OGBENCH_IMPLS', '/project/6067317/jwquan/ogbench/impls')
sys.path.insert(0, IMPLS)
os.chdir(IMPLS)  # main.py resolves --agent=agents/X.py relative to impls/

import main as ogbench_main  # noqa: E402
from absl import app  # noqa: E402

_orig_setup_wandb = ogbench_main.setup_wandb


def _setup_wandb_patched(*args, **kwargs):
    kwargs['mode'] = os.environ.get('WMPP_WANDB_MODE', 'offline')
    return _orig_setup_wandb(*args, **kwargs)


ogbench_main.setup_wandb = _setup_wandb_patched

if __name__ == '__main__':
    app.run(ogbench_main.main)
