"""Generic, reusable plumbing shared across entrypoints — no orchestration or science.

Holds the canonical repo/data paths and small helpers (git sha, run id, durable JSONL
writes) so the orchestrator and any future entrypoint share one implementation.
"""
import json
import os
import subprocess
from datetime import datetime
from typing import TextIO

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")


def git_sha() -> str:
    """Short HEAD sha of the repo, or 'nogit' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "nogit"


def make_run_id(sha: str) -> str:
    """A unique, sortable run id: <timestamp>_<git_sha>."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + sha


def write_rows(jf: TextIO, rows: list[dict]) -> None:
    """Append rows to a durable JSONL log and fsync (a crash keeps finished tasks)."""
    for r in rows:
        jf.write(json.dumps(r) + "\n")
    jf.flush()
    os.fsync(jf.fileno())
