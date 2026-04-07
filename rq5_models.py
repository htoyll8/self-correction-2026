#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import math

import numpy as np
import pandas as pd


def find_iter_cols(cols) -> List[Tuple[int, str]]:
    out = []
    for c in cols:
        if isinstance(c, str) and c.startswith("pass_iter_"):
            try:
                t = int(c.split("pass_iter_")[1])
                out.append((t, c))
            except Exception:
                pass
    out.sort(key=lambda x: x[0])
    return out


def normalize_curve(vals: List[float]) -> List[float]:
    # Treat as percentages if values look like 0..100
    vmax = max(vals) if vals else 0.0
    if vmax > 1.0 + 1e-9 and vmax <= 100.0 + 1e-9:
        vals = [v / 100.0 for v in vals]
    # clamp to [0,1]
    return [min(1.0, max(0.0, float(v))) for v in vals]


def spearman_rho(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None
    if np.all(y == y[0]):
        return None
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    denom = np.sqrt((rxm**2).sum()) * np.sqrt((rym**2).sum())
    if denom == 0:
        return None
    return float((rxm * rym).sum() / denom)


def linear_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None
    if np.all(x == x[0]):
        return None
    return float(np.polyfit(x, y, 1)[0])


def first_stable_iter(curve: np.ndarray, eps: float = 1e-3, window: int = 2) -> Optional[int]:
    """
    Heuristic: first iteration t such that for all i>=t,
    |curve[i] - curve[i-1]| <= eps for 'window' consecutive steps (default 2).
    """
    if len(curve) < 3:
        return None
    diffs = np.abs(np.diff(curve))
    # need window consecutive small diffs starting at t
    for t in range(1, len(curve) - window):
        if np.all(diffs[t : t + window] <= eps):
            return int(t + 1)  # stable starting at this iteration index
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "Desktop" / "self_corrections_pandas.csv",
        help="Aggregated CSV with pass_iter_* columns",
    )
    ap.add_argument("--out", type=Path, default=Path("rq5_group_curve_metrics.csv"))
    ap.add_argument("--eps", type=float, default=1e-3, help="stability threshold")
    ap.add_argument("--window", type=int, default=2, help="stability consecutive window")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    iter_cols = find_iter_cols(df.columns)
    if not iter_cols:
        raise ValueError("No pass_iter_* columns found.")

    iters = np.array([t for t, _ in iter_cols], dtype=float)

    rows = []
    for _, r in df.iterrows():
        curve_raw = []
        for _, c in iter_cols:
            v = r.get(c)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                curve_raw.append(0.0)
            else:
                if isinstance(v, str):
                    v = v.strip()
                    if v.endswith("%"):
                        v = v[:-1]          # remove %
                        curve_raw.append(float(v) / 100.0)
                    else:
                        curve_raw.append(float(v))
                else:
                    curve_raw.append(float(v))

        curve = np.array(normalize_curve(curve_raw), dtype=float)

        rho = spearman_rho(iters, curve)
        slope = linear_slope(iters, curve)
        auc = float(np.trapz(curve, dx=1.0))  # sum-ish; max ~ (#iters) if curve near 1
        fst = first_stable_iter(curve, eps=args.eps, window=args.window)

        out = {
            "dataset": r.get("dataset", ""),
            "difficulty": r.get("difficulty", ""),
            "history": int(r.get("history", 0)) if not pd.isna(r.get("history", 0)) else 0,
            "model": r.get("model", ""),

            # carry through your existing headline metrics if present
            "initial_pass_rate": r.get("initial_pass_rate", np.nan),
            "recovery_rate": r.get("recovery_rate", np.nan),
            "avg_attempts_recovered": r.get("avg_attempts_recovered", np.nan),

            # curve-derived metrics
            "spearman_rho_iter_pass_mean_curve": rho,
            "linear_slope_per_iter_mean_curve": slope,
            "auc_trapz_mean_curve": auc,
            "first_stable_iter_mean_curve": fst,
        }
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print(f"Wrote: {args.out}")
    # show a quick peek
    show_cols = [
        "dataset", "difficulty", "history", "model",
        "spearman_rho_iter_pass_mean_curve",
        "linear_slope_per_iter_mean_curve",
        "auc_trapz_mean_curve",
        "first_stable_iter_mean_curve",
    ]
    print(out_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
