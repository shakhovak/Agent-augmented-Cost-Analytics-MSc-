from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from cost_agent_mvp.viz.chart_specs import SERVICE_COST_COLUMNS

PRICE_PER_ACCOUNT_DAY_USD = 100.0
COST_CAP_USD = 30.0  # adjust as needed


def _ensure_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col]).dt.date
    return out


def _service_total(df: pd.DataFrame) -> pd.Series:
    # Only the 3 included service columns (dialog excluded)
    return df[list(SERVICE_COST_COLUMNS)].sum(axis=1)


def filter_active_service_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where at least one included service has cost > 0.
    This removes total_cost == 0 rows AND removes dialog-only rows.
    """
    if df.empty:
        return df.copy()
    s = _service_total(df)
    return df.loc[s > 0].copy()


def active_users(df_day: pd.DataFrame) -> float:
    df = filter_active_service_rows(df_day)
    return float(df["account_id"].nunique())


def dials_analyzed(df_day: pd.DataFrame) -> float:
    """
    Your definition: count of non-zero service lines
    = number of rows where any included service cost > 0.
    """
    df = filter_active_service_rows(df_day)
    return float(len(df))


def total_cost_services(df_day: pd.DataFrame) -> float:
    df = filter_active_service_rows(df_day)
    if df.empty:
        return 0.0
    return float(_service_total(df).sum())


def avg_cost_per_account_non_diluted(df_day: pd.DataFrame) -> float:
    """
    "Non-diluted avg" (like your real dashboard):
    For each service separately:
      mean over accounts with service_cost > 0 of (per-account mean(service_cost))
    Then sum the 3 service means.
    """
    df = filter_active_service_rows(df_day)
    if df.empty:
        return 0.0

    total = 0.0
    for col in SERVICE_COST_COLUMNS:
        sub = df[df[col] > 0]
        if sub.empty:
            continue
        per_acc = sub.groupby("account_id", as_index=True)[col].mean()
        total += float(per_acc.mean())
    return float(total)


def per_account_cost_distribution(df_day: pd.DataFrame) -> pd.Series:
    """
    Distribution used in plots: total service cost per account-day (USD).
    """
    return per_account_day_costs(df_day)


@dataclass(frozen=True)
class ChurnResult:
    new_users: int
    churned_users: int
    net_change: int


def user_churn(df_today: pd.DataFrame, df_yesterday: pd.DataFrame) -> ChurnResult:
    today = filter_active_service_rows(df_today)
    yest = filter_active_service_rows(df_yesterday)

    users_today = set(today["account_id"].unique())
    users_yest = set(yest["account_id"].unique())

    new_users = len(users_today - users_yest)
    churned = len(users_yest - users_today)
    net = len(users_today) - len(users_yest)

    return ChurnResult(new_users=new_users, churned_users=churned, net_change=net)


def service_combo_counts(df_day: pd.DataFrame) -> dict[str, int]:
    """
    Plot D categories (3-service world):

    - Has 1 Service
    - Has 2 Services
    - All 3 Services
    Based on included services only.
    """
    df = df_day.copy()
    if df.empty:
        return {
            # "Only Classifications": 0,
            "Has 1 Service": 0,
            "Has 2 Services": 0,
            "All 3 Services": 0,
        }

    # Build per-account flags for whether the account used each service today
    active = filter_active_service_rows(df)
    if active.empty:
        return {
            # "Only Classifications": 0,
            "Has 1 Service": 0,
            "Has 2 Services": 0,
            "All 3 Services": 0,
        }

    g = active.groupby("account_id", as_index=True)
    has_tasks = g["total_cost_tasks"].sum() > 0
    has_class = g["total_cost_classifications"].sum() > 0
    has_amocrm = g["cost_amocrm_call"].sum() > 0

    n_services = has_tasks.astype(int) + has_class.astype(int) + has_amocrm.astype(int)

    # only_class = int(((has_class) & (~has_tasks) & (~has_amocrm)).sum())
    has_1 = int((n_services == 1).sum())
    has_2 = int((n_services == 2).sum())
    all_3 = int((n_services == 3).sum())

    return {
        # "Only Classifications": only_class,
        "Has 1 Service": has_1,
        "Has 2 Services": has_2,
        "All 3 Services": all_3,
    }


def build_daily_series(df_all: pd.DataFrame, dates: Iterable[date]) -> pd.DataFrame:
    """
    Build a daily table for trends.
    """
    df_all = _ensure_date_col(df_all)
    rows: list[dict[str, object]] = []
    for d in dates:
        day_df = df_all[df_all["date"] == d]
        rows.append(
            {
                "date": d,
                "total_cost_services": total_cost_services(day_df),
                "dials_analyzed": dials_analyzed(day_df),
                "active_users": active_users(day_df),
                "avg_cost_per_active_account_day": avg_cost_per_active_account_day(day_df),
                "p95_cost_per_account_day": pxx_cost_per_account_day(day_df, 0.95),
                "pct_accounts_above_cap": pct_accounts_above_cost_cap(day_df),
                "avg_margin_per_account_day": avg_margin_per_account_day(day_df),
            }
        )
    return pd.DataFrame(rows)


def delta(curr: float, prev: float) -> tuple[float, float]:
    """
    Returns (abs_delta, pct_delta).
    pct_delta is 0.0 when prev == 0 to avoid division errors.
    """
    abs_delta = float(curr) - float(prev)
    if float(prev) == 0.0:
        return abs_delta, 0.0
    return abs_delta, abs_delta / float(prev)


def per_account_day_costs(df_day: pd.DataFrame) -> pd.Series:
    """
    Cost per account for the day (USD), based on included service columns only.
    Returns a Series indexed by account_id with total service cost for that account-day.
    Accounts with 0 total service cost are excluded (active service users only).
    """
    df = filter_active_service_rows(df_day)
    if df.empty:
        return pd.Series(dtype=float)

    # Sum per row then sum across rows per account (handles multiple rows per account)
    row_total = _service_total(df)
    per_acc = row_total.groupby(df["account_id"]).sum()
    per_acc = per_acc[per_acc > 0].sort_values()
    per_acc.name = "cost_per_account_day"
    return per_acc


def avg_cost_per_active_account_day(df_day: pd.DataFrame) -> float:
    """
    Diluted average: total_cost_services / active_users.
    This is the one that reconciles with total cost.
    """
    au = active_users(df_day)
    if au == 0.0:
        return 0.0
    return float(total_cost_services(df_day) / au)


def pxx_cost_per_account_day(df_day: pd.DataFrame, q: float = 0.95) -> float:
    """
    Percentile of per-account-day costs among active service users (USD).
    q=0.95 -> p95.
    """
    s = per_account_day_costs(df_day)
    if s.empty:
        return 0.0
    return float(s.quantile(q))


def pct_accounts_above_cost_cap(df_day: pd.DataFrame, cap_usd: float = COST_CAP_USD) -> float:
    """
    Share of active service accounts whose per-account-day cost exceeds cap_usd.
    Returned as fraction (0..1).
    """
    s = per_account_day_costs(df_day)
    if s.empty:
        return 0.0
    return float((s > float(cap_usd)).mean())


def avg_margin_per_account_day(
    df_day: pd.DataFrame,
    price_usd: float = PRICE_PER_ACCOUNT_DAY_USD,
) -> float:
    """
    Average margin per active service account-day:
      margin = price_usd - avg_cost_per_active_account_day
    """
    return float(price_usd) - float(avg_cost_per_active_account_day(df_day))
