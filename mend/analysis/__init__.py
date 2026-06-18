"""Analysis layer for the canonical results table.

`results` writes the table (and its derived figure/metrics/table); `metrics` reads it back
and computes every reported number. The two are kept in sync via metrics_definitions.md.
"""
from mend.analysis import metrics, results

__all__ = ["metrics", "results"]
