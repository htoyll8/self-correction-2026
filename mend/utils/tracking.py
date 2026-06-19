"""MLflow experiment-tracking glue: a local SQLite backend plus helpers to log our
canonical metric shape to the active run.
"""
import os

import mlflow
from mlflow.exceptions import MlflowException

from mend.utils import REPO

MLFLOW_DB = "sqlite:///" + os.path.join(REPO, "mlflow.db")
MLFLOW_ARTIFACTS = "file:" + os.path.join(REPO, "mlartifacts")
MLFLOW_EXPERIMENT = "self-correction-rerun"

# USD per 1M tokens (input, output). Update if OpenAI/Anthropic pricing changes.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-4-0613": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


def setup_mlflow() -> None:
    """Use a SQLite tracking backend (MLflow's recommended local store) with a dedicated
    artifacts dir; create the experiment on first run."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    try:  # EAFP: create once; swallow only "already exists", re-raise real errors
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS)
    except MlflowException as e:
        if "already exists" not in str(e).lower():
            raise
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_condition_metrics(m: dict) -> None:
    """Log the (single-condition) headline scalars to the active MLflow run."""
    mm = next(iter(m.values()), None)
    if mm is None:
        return
    for key in ("initial_pass_rate", "recovered_of_failed", "recovered_of_all", "task_solve_rate"):
        mlflow.log_metric(key, mm[key] or 0.0)
    for rk, rv in mm["rounds"].items():
        val = rv["mean_pass_fraction"] if isinstance(rv, dict) else rv
        if val is not None:
            mlflow.log_metric(rk, val)


def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """USD estimate from PRICES; None if the model isn't listed (log tokens but not cost)."""
    price = PRICES.get(model_name)
    if price is None:
        return None
    price_in, price_out = price
    return prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out


def log_token_usage(model_name: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Log token totals (and a cost estimate if the model is priced) to the active run.
    Returns the estimated USD cost, or None if the model isn't in PRICES."""
    mlflow.log_metric("prompt_tokens", prompt_tokens)
    mlflow.log_metric("completion_tokens", completion_tokens)
    mlflow.log_metric("total_tokens", prompt_tokens + completion_tokens)
    cost = estimate_cost(model_name, prompt_tokens, completion_tokens)
    if cost is not None:
        mlflow.log_metric("estimated_cost_usd", cost)
    return cost
