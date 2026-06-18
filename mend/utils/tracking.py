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
