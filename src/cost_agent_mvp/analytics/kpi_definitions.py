"""Deterministic metric functions (active_users, dials, etc.)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from src.core.errors import ValidationError

# -----------------------------
# Helpers
# -----------------------------


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns for KPI computation: {missing}")


def _safe_float(v) -> float:
    if v is None:
        return 0.0
    try:
        if pd.isna(v):
            return 0.0
    except Exception:
        pass
    return float(v)


def _quantile(s: pd.Series, q: float) -> float:
    if s.empty:
        return 0.0
    return float(s.quantile(q))


# -----------------------------
# Core KPI computations
# -----------------------------


def active_users(df: pd.DataFrame) -> int:
    """
    Count distinct account_id with total_cost > 0.
    """
    _require_columns(df, ["account_id", "total_cost"])
    tmp = df[df["total_cost"].fillna(0) > 0]
    return int(tmp["account_id"].nunique())


def dials_analyzed(df: pd.DataFrame) -> int:
    """
    Usage proxy: count distinct (account_id, chat_id, chat_type) with total_cost > 0.
    """
    _require_columns(df, ["account_id", "chat_id", "chat_type", "total_cost"])
    tmp = df[df["total_cost"].fillna(0) > 0]
    if tmp.empty:
        return 0
    return int(tmp.drop_duplicates(subset=["account_id", "chat_id", "chat_type"]).shape[0])


def total_cost_sum(df: pd.DataFrame) -> float:
    _require_columns(df, ["total_cost"])
    return float(df["total_cost"].fillna(0).sum())


def component_costs_sum(df: pd.DataFrame) -> dict[str, float]:
    """
    Returns key component sums used in dashboards and explanations.
    """
    cols = [
        "total_cost_tasks",
        "total_cost_classifications",
        "cost_dialog",
        "cost_task",
        "cost_classification",
        "cost_qc",
        "cost_check_list",
    ]
    out: dict[str, float] = {}
    for c in cols:
        if c in df.columns:
            out[c] = float(df[c].fillna(0).sum())
    return out


def avg_cost_per_account_diluted(df: pd.DataFrame) -> float:
    """
    Diluted average: total_cost_sum / active_users.
    """
    tot = total_cost_sum(df)
    au = active_users(df)
    return float(tot / au) if au > 0 else 0.0


def avg_cost_per_account_non_diluted(df: pd.DataFrame) -> float:
    """
    Non-diluted average (as discussed for your dashboard):
    - avg over accounts that have tasks: mean(sum(total_cost_tasks) per account)
    - avg over accounts that have classifications: mean(sum(total_cost_classifications) per account)
    - return sum of both means

    This may differ from total_cost_sum/active_users. Keep definition consistent for fidelity.
    """
    _require_columns(
        df,
        [
            "account_id",
            "total_cost_tasks",
            "total_cost_classifications",
            "has_tasks",
            "has_classifications",
        ],
    )

    # Tasks part
    df_t = df[df["has_tasks"].fillna(0).astype(int) == 1]
    tasks_mean = 0.0
    if not df_t.empty:
        per_acc_tasks = df_t.groupby("account_id")["total_cost_tasks"].sum()
        tasks_mean = float(per_acc_tasks.mean()) if not per_acc_tasks.empty else 0.0

    # Classifications part
    df_c = df[df["has_classifications"].fillna(0).astype(int) == 1]
    cls_mean = 0.0
    if not df_c.empty:
        per_acc_cls = df_c.groupby("account_id")["total_cost_classifications"].sum()
        cls_mean = float(per_acc_cls.mean()) if not per_acc_cls.empty else 0.0

    return float(tasks_mean + cls_mean)


def per_account_total_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-account total cost for the given df scope (typically one day).
    Columns: account_id, total_cost
    """
    _require_columns(df, ["account_id", "total_cost"])
    out = (
        df.groupby("account_id", as_index=False)["total_cost"]
        .sum()
        .rename(columns={"total_cost": "total_cost"})
    )
    return out


def distribution_stats(values: pd.Series) -> dict[str, float]:
    """
    Compact distribution summary for reporting/evaluation.
    """
    s = values.fillna(0).astype(float)
    if s.empty:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "std": 0.0,
        }

    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "p75": _quantile(s, 0.75),
        "p90": _quantile(s, 0.90),
        "p95": _quantile(s, 0.95),
        "std": float(s.std(ddof=0)),
    }


def histogram_table(values: pd.Series, bins: int = 20) -> pd.DataFrame:
    """
    Returns a compact histogram table: bin_left, bin_right, count.
    """
    s = values.fillna(0).astype(float)
    if s.empty:
        return pd.DataFrame(columns=["bin_left", "bin_right", "count"])

    # If all values equal, produce one bin
    vmin = float(s.min())
    vmax = float(s.max())
    if vmin == vmax:
        return pd.DataFrame([{"bin_left": vmin, "bin_right": vmax, "count": int(len(s))}])

    counts, edges = np.histogram(s.values, bins=bins)
    rows = []
    for i in range(len(counts)):
        rows.append(
            {
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "count": int(counts[i]),
            }
        )
    return pd.DataFrame(rows)


def delta(current: float, previous: float) -> tuple[float, float]:
    """
    Returns (delta_abs, delta_pct). If previous is 0, pct is 0 unless current>0, then 1.0.
    """
    cur = float(current)
    prev = float(previous)
    d_abs = cur - prev
    if prev == 0.0:
        d_pct = 0.0 if cur == 0.0 else 1.0
    else:
        d_pct = d_abs / prev
    return float(d_abs), float(d_pct)
