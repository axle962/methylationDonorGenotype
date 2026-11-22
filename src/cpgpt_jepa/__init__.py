"""Latent block prediction on frozen CpGPT embeddings (docs/proposal.md)."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def finish(ok: bool, reason: str = "") -> None:
    """Write the stage's sanity verdict for slurm/common.sh (which prints it last) and exit."""
    verdict = "SANITY_OK" if ok else f"SANITY_FAIL: {reason}"
    sanity_file = os.environ.get("SANITY_FILE")
    if sanity_file:
        Path(sanity_file).write_text(verdict + "\n")
    else:
        print(verdict)
    sys.exit(0 if ok else 1)
