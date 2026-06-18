"""
Strategy registry.

A strategy maps (model, description, scorer, np, max_attempts) to a list of scored
Attempts. Add one (e.g. an anti-unification-history refiner — see
Model.generate_antiunified_history) by writing a module with `run(...)` and registering
it here under the name used as --refine_mode.
"""
from collections.abc import Callable

from mend.strategies.base import Attempt, Scorer, extract_code
from mend.strategies import critique_refine

# name (the --refine_mode value) -> strategy callable
Strategy = Callable[..., list[Attempt]]
STRATEGIES: dict[str, Strategy] = {
    "critique+refine": critique_refine.run,
}


def get_strategy(name: str) -> Strategy:
    """Look up a registered strategy, failing loudly on an unknown name."""
    if name not in STRATEGIES:
        raise ValueError(f"refine_mode {name!r} not supported; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[name]
