"""Evaluator layer: turn a task's test material into a scorer callable.

The scorer runs candidate programs in an isolated subprocess (test_worker.py) and returns
the fraction of tests passed.
"""
from mend.evaluators.scorer import make_scorer

__all__ = ["make_scorer"]
