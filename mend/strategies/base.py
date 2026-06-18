"""
Shared types and helpers for refinement strategies.

A strategy is a policy that, given a model, a task description, and a scorer, produces a
list of scored `Attempt`s — the full seed/refinement trajectory for one task. The
orchestrator turns those into canonical rows; strategies stay ignorant of run metadata,
logging, and storage.
"""
import re
from collections.abc import Callable
from dataclasses import dataclass

# A scorer maps a candidate program to the fraction of tests it passes, in [0, 1].
Scorer = Callable[[str], float]


@dataclass
class Attempt:
    """One scored program in a seed's trajectory."""
    seed_idx: int
    attempt: int             # 0 = initial seed; 1..K = refinement rounds
    program: str
    feedback: str | None     # critique that produced this program (None for the seed)
    pass_fraction: float
    passed: bool


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of a model response (or fall back to raw text)."""
    m = re.findall(r"```(?:\w+)?\n?(.*?)```", text or "", flags=re.DOTALL)
    return (m[0] if m else (text or "")).strip()
