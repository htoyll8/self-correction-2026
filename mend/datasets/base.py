"""
Benchmark-agnostic task representation shared by every dataset loader.
A loader turns a raw benchmark example into a `Task`. The evaluator scores a candidate by running `setup`, then the candidate program, then `prelude` (e.g. binding `candidate` for HumanEval), then each statement in `tests` (one assert each, for partial credit).
"""
from dataclasses import dataclass

from datasets import Dataset, load_dataset


@dataclass
class Task:
    """One benchmark problem, normalized across datasets."""
    task_id: str
    description: str          # the natural-language prompt shown to the model
    setup: str               # code executed before the candidate program
    tests: list[str]         # individual assert statements (scored for partial credit)
    prelude: str = ""        # code executed after the program (e.g. bind `candidate`)

    @property
    def n_tests(self) -> int:
        return len(self.tests)


def take(source: str, n_tasks: int, split: str = "test") -> Dataset:
    """Load the first `n_tasks` rows of a HF dataset split, lazily.

    `ds.select` avoids materializing the whole split into a Python list before slicing,
    which matters for large benchmarks (e.g. APPS).
    """
    ds = load_dataset(source)[split]
    return ds.select(range(min(n_tasks, ds.num_rows)))
