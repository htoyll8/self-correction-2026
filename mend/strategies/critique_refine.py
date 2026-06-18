"""
Critique-then-refine strategy (the original study's loop).

Generate `np` seeds at temperature 0.7. For each seed that fails, loop: ask the model to
critique the current program, refine it from that critique, and re-score — until it passes
or `max_attempts` rounds are spent. Refinement steps run at temperature 0.
"""
from mend.models.llm import Model
from mend.strategies.base import Attempt, Scorer, extract_code


def run(model: Model, description: str, scorer: Scorer, np_: int, max_attempts: int) -> list[Attempt]:
    attempts: list[Attempt] = []
    seeds = [extract_code(s) for s in model.generate(task_description=description, n=np_, temperature=0.7)]
    for si, seed in enumerate(seeds):
        frac = scorer(seed)
        passed = frac == 1.0
        attempts.append(Attempt(si, 0, seed, None, frac, passed))
        if passed:
            continue
        cur = seed
        for k in range(1, max_attempts + 1):
            feedback = model.generate_feedback(description, cur, temperature=0)
            cur = extract_code(model.refine(description, cur, feedback, temperature=0))
            frac = scorer(cur)
            passed = frac == 1.0
            attempts.append(Attempt(si, k, cur, feedback, frac, passed))
            if passed:
                break
    return attempts
