"""Deterministic metric functions (active_users, dials, etc.)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from cost_agent_mvp.core.errors import ValidationError

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
        "cost_dialog",
        "total_cost_tasks",
        "total_cost_classifications",
        "cost_amocrm_call",  # <-- add this
        # keep detailed subcomponents if you want to expose them too
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


def avg_cost_per_active_service_account(df: pd.DataFrame) -> float:
    """
    Average cost per *service-active* account (not diluted by all active accounts).

    Definition (v0, deterministic):
      - tasks_mean  = mean over accounts with has_tasks==1 of sum(total_cost_tasks)
      - cls_mean    = mean over accounts with has_classifications==1 of sum(total_cost_classifications)
      - result      = tasks_mean + cls_mean

    Why: distinguishes “higher cost because more accounts used a service”
    from “higher cost because spend per service-active account increased”.

    Note: this is intentionally NOT equal to total_cost_sum/active_users.
    """
    return avg_cost_per_account_non_diluted(df)


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


def delta_abs(current: float, previous: float) -> float:
    return float(current) - float(previous)


def delta_pct(current: float, previous: float) -> float:
    cur = float(current)
    prev = float(previous)
    if prev == 0.0:
        return 0.0 if cur == 0.0 else 1.0
    return (cur - prev) / prev


def delta(current: float, previous: float) -> tuple[float, float]:
    return delta_abs(current, previous), delta_pct(current, previous)


def active_account_days(df: pd.DataFrame) -> int:
    """
    Count distinct (account_id, date) where total_cost > 0.

    Why: in sparse/zero-inflated data this is a more stable activity KPI than
    raw row counts. It also supports explaining spikes as “more active days”.
    """
    _require_columns(df, ["account_id", "date", "total_cost"])
    tmp = df[df["total_cost"].fillna(0) > 0]
    if tmp.empty:
        return 0
    return int(tmp.drop_duplicates(subset=["account_id", "date"]).shape[0])


def avg_cost_per_active_account_day(df: pd.DataFrame) -> float:
    """
    Normalized spend: total_cost_sum / active_account_days.
    """
    tot = total_cost_sum(df)
    aad = active_account_days(df)
    return float(tot / aad) if aad > 0 else 0.0


def component_costs_sum_top_level(df: pd.DataFrame) -> dict[str, float]:
    """
    Sum ONLY top-level cost components (no double counting).

    Intended for service mix / contribution charts:
      - cost_dialog
      - total_cost_tasks
      - total_cost_classifications
      - cost_amocrm_call

    Notes:
      - Some datasets may omit cost_amocrm_call; missing columns are treated as 0.
      - If df is empty, returns zeros.
    """
    top_cols = [
        "cost_dialog",
        "total_cost_tasks",
        "total_cost_classifications",
        "cost_amocrm_call",
    ]

    if df.empty:
        return dict.fromkeys(top_cols, 0.0)

    missing = [c for c in top_cols if c not in df.columns]
    # Treat missing as zeros rather than error — keeps evidence pack robust across variants
    present = [c for c in top_cols if c in df.columns]

    sums = {c: float(df[c].fillna(0).sum()) for c in present}
    for c in missing:
        sums[c] = 0.0
    return sums


def service_mix_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Service/component cost mix as a table (TOP-LEVEL components only, no double counting).

    Output columns:
      - component: str (e.g., cost_dialog, total_cost_tasks, ...)
      - cost: float
      - share: float in [0,1] (0 if total is 0)
    """
    comp = component_costs_sum_top_level(df)
    total = float(sum(comp.values()))

    rows: list[dict[str, object]] = []
    for k, v in sorted(comp.items()):
        rows.append(
            {
                "component": str(k),
                "cost": float(v),
                "share": float(v / total) if total > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows, columns=["component", "cost", "share"])


def rows_per_account_day_stats(df: pd.DataFrame, cap: int | None = None) -> dict[str, float]:
    """
    Row-density diagnostics: how many rows exist per (account_id, date).

    Useful for cap / concurrency spike explanations.

    Returns a compact stats dict:
      - mean, median, p95, p99, max
      - n_account_days
      - n_over_cap (if cap provided, else 0)
      - pct_over_cap (if cap provided, else 0.0)
    """
    _require_columns(df, ["account_id", "date"])
    if df.empty:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "n_account_days": 0.0,
            "n_over_cap": 0.0,
            "pct_over_cap": 0.0,
        }

    counts = (
        df.groupby(["account_id", "date"], as_index=False).size().rename(columns={"size": "n_rows"})
    )
    s = counts["n_rows"].astype(float)

    n_days = float(len(s))
    n_over = 0.0
    pct_over = 0.0
    if cap is not None:
        n_over = float((s > float(cap)).sum())
        pct_over = float(n_over / n_days) if n_days > 0 else 0.0

    return {
        "mean": float(s.mean()) if not s.empty else 0.0,
        "median": float(s.median()) if not s.empty else 0.0,
        "p95": _quantile(s, 0.95),
        "p99": _quantile(s, 0.99),
        "max": float(s.max()) if not s.empty else 0.0,
        "n_account_days": n_days,
        "n_over_cap": n_over,
        "pct_over_cap": pct_over,
    }


def top_accounts(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top-N accounts by total_cost within the given df scope.

    Output columns:
      - account_id
      - total_cost
      - share_of_total_cost (fraction of total_cost_sum)
      - rank (1..N)
    """
    if n <= 0:
        raise ValidationError("n must be positive for top_accounts().")

    per_acc = per_account_total_cost(df)
    if per_acc.empty:
        return pd.DataFrame(columns=["rank", "account_id", "total_cost", "share_of_total_cost"])

    total = total_cost_sum(df)
    out = per_acc.sort_values("total_cost", ascending=False, kind="mergesort").head(n).copy()
    out["share_of_total_cost"] = out["total_cost"].apply(
        lambda v: float(v / total) if total > 0 else 0.0
    )
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)
