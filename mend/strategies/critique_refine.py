"""Critique-then-refine strategy (the original study's loop and prompts).

Generate `np` seeds at temperature 0.7. For each failing seed, loop: critique the current
program, refine it from that critique, and re-score — until it passes or `max_attempts`
rounds are spent. Refinement runs at temperature 0.

The prompt wording below is the study's method (moved verbatim from the legacy Model class)
and is pinned by tests/test_prompts.py so the rerun's parity claim is enforced.
"""
from mend.models.llm import Model
from mend.strategies.base import Attempt, Scorer, extract_code


def _seed_prompt(description: str) -> str:
    return (
        description +
        "Please provide Python code wrapped in triple backticks like:\n"
        "```python\n"
        "# your code here\n"
        "```\n\n"
    )


def _critique_prompt(description: str, program: str) -> str:
    has_history = "Summary of previous attempts:" in program
    section_label = (
        "Context (previous attempts and current program)"
        if has_history else
        "Program to Critique"
    )
    return (
        f"The following attempt did not pass all of its tests.\n\n"
        f"Please explain what might be wrong.\n\n"
        f"Task:\n{description}\n\n"
        f"{section_label}:\n{program}\n\n"
    )


def _refine_prompt(description: str, program: str, feedback: str) -> str:
    return (
        f"Task:\n{description}\n\n"
        f"Current Program:\n{program}\n\n"
        f"Feedback:\n{feedback}\n\n"
        f"Revise the program to address the feedback. "
        f"Only return the corrected code."
    )


def run(model: Model, description: str, scorer: Scorer, np_: int, max_attempts: int) -> list[Attempt]:
    attempts: list[Attempt] = []
    seeds = [extract_code(model.complete(_seed_prompt(description), temperature=0.7)) for _ in range(np_)]
    for si, seed in enumerate(seeds):
        frac = scorer(seed)
        passed = frac == 1.0
        attempts.append(Attempt(si, 0, seed, None, frac, passed))
        if passed:
            continue
        cur = seed
        for k in range(1, max_attempts + 1):
            feedback = model.complete(_critique_prompt(description, cur), temperature=0)
            cur = extract_code(model.complete(_refine_prompt(description, cur, feedback), temperature=0))
            frac = scorer(cur)
            passed = frac == 1.0
            attempts.append(Attempt(si, k, cur, feedback, frac, passed))
            if passed:
                break
    return attempts
