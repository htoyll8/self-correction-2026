"""Dataset registry.

Add a benchmark by writing a loader module that exposes `load(n_tasks) -> list[Task]`
and registering it in LOADERS below. Nothing else in the framework needs to change.
"""
from mend.datasets.base import Task
from mend.datasets import mbppplus, humaneval

LOADERS = {
    "mbppplus": mbppplus.load,
    "humaneval": humaneval.load,
}


def load_tasks(name: str, n_tasks: int) -> list[Task]:
    """Load the first `n_tasks` of a registered benchmark as normalized Tasks."""
    if name not in LOADERS:
        raise ValueError(f"dataset {name!r} not supported; choose from {sorted(LOADERS)}")
    return LOADERS[name](n_tasks)
