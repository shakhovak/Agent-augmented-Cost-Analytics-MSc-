#!/usr/bin/env python3
"""
Generate factual (ground-truth) instances from a joint_costs_daily CSV.

For each question type, sample N cases and compute the numeric answer(s)
directly from the CSV. Output as JSONL for downstream question generation
and evaluation.

CSV columns expected (subset used here):
- account_id, date, total_cost, cost_dialog, total_cost_tasks,
  total_cost_classifications, cost_amocrm_call
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- Service components mapping (aligned to your CSV schema) ---
SERVICE_COMPONENTS = [
    ("dialog", "cost_dialog"),
    ("tasks", "total_cost_tasks"),
    ("classifications", "total_cost_classifications"),
    ("amocrm_call", "cost_amocrm_call"),
]

# ---- Constants (keep aligned with kpi_definitions) ----
PRICE_PER_ACCOUNT_DAY_USD = 100.0  # :contentReference[oaicite:8]{index=8}
COST_CAP_USD = 30.0  # :contentReference[oaicite:9]{index=9}

SERVICE_ONLY_COLS = [
    "total_cost_tasks",
    "total_cost_classifications",
    "cost_amocrm_call",
]
# ------------------------- Utilities -------------------------


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {
        "account_id",
        "date",
        "total_cost",
        "cost_dialog",
        "total_cost_tasks",
        "total_cost_classifications",
        "cost_amocrm_call",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Parse date: keep as python date for stable JSON and comparisons.
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.date

    # Ensure numerics
    for col in [
        "total_cost",
        "cost_dialog",
        "total_cost_tasks",
        "total_cost_classifications",
        "cost_amocrm_call",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["account_id"] = pd.to_numeric(df["account_id"], errors="raise").astype("int64")
    return df


def _to_iso(d: date) -> str:
    return d.isoformat()


def _sample_values(values: list[Any], n: int, rng: np.random.Generator) -> list[Any]:
    """Sample n items from values, without replacement if possible."""
    if not values:
        raise ValueError("Cannot sample from empty list.")
    if n <= len(values):
        idx = rng.choice(len(values), size=n, replace=False)
    else:
        idx = rng.choice(len(values), size=n, replace=True)
    return [values[i] for i in idx]


def _candidate_days(df: pd.DataFrame) -> list[date]:
    """Prefer days with some spend, to avoid trivial all-zero answers."""
    day_sum = df.groupby("date", as_index=True)["total_cost"].sum()
    days = day_sum[day_sum > 0].index.tolist()
    # Fallback: if everything is zero (unlikely), allow all days.
    if not days:
        days = sorted(df["date"].unique().tolist())
    return sorted(days)


def _anchor_day_from_df(df: pd.DataFrame) -> date:
    return max(df["date"])


def _component_totals_for_day(df: pd.DataFrame, day: date) -> dict[str, float]:
    sub = df[df["date"] == day]
    totals: dict[str, float] = {}
    for comp_name, col in SERVICE_COMPONENTS:
        totals[comp_name] = float(sub[col].sum())
    totals["_services_total"] = float(sum(totals.values()))
    return totals


def _per_account_day_cost(df: pd.DataFrame, day: date) -> pd.Series:
    """Sum total_cost across chats/types -> account-day total cost."""
    sub = df[df["date"] == day]
    by_acc = sub.groupby("account_id", as_index=True)["total_cost"].sum()
    return by_acc


# ------------------------- Fact types -------------------------


def sample_total_cost_single_day(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    tol_abs: float = 0.01,
) -> list[dict[str, Any]]:
    days = _sample_values(_candidate_days(df), n, rng)
    out: list[dict[str, Any]] = []
    for d in days:
        val = float(df.loc[df["date"] == d, "total_cost"].sum())
        out.append(
            {
                "type": "total_cost_single_day",
                "params": {"day": _to_iso(d)},
                "targets": [{"key": "sum_total_cost", "value": val, "tol_abs": tol_abs}],
                # Allowed numbers for question phrasing (date only here)
                "allowed_numbers": [_to_iso(d)],
            }
        )
    return out


def sample_total_cost_window(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    window_min_days: int = 3,
    window_max_days: int = 14,
    tol_abs: float = 0.01,
) -> list[dict[str, Any]]:
    days = _candidate_days(df)
    # Work in sorted day list so we can sample contiguous windows that exist in data.
    out: list[dict[str, Any]] = []
    if len(days) < window_min_days:
        raise ValueError("Not enough unique days to sample windows.")

    for _ in range(n):
        w = int(rng.integers(window_min_days, window_max_days + 1))
        start_idx = int(rng.integers(0, max(1, len(days) - w)))
        start = days[start_idx]
        end = days[min(start_idx + w - 1, len(days) - 1)]

        mask = (df["date"] >= start) & (df["date"] <= end)
        val = float(df.loc[mask, "total_cost"].sum())
        out.append(
            {
                "type": "total_cost_window",
                "params": {"start": _to_iso(start), "end": _to_iso(end)},
                "targets": [{"key": "sum_total_cost", "value": val, "tol_abs": tol_abs}],
                "allowed_numbers": [_to_iso(start), _to_iso(end)],
            }
        )
    return out


def sample_top_service_by_cost(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    tol_abs: float = 0.01,
) -> list[dict[str, Any]]:
    # We may need retries if a sampled day has zero in all service components.
    days = _candidate_days(df)
    out: list[dict[str, Any]] = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        d = days[int(rng.integers(0, len(days)))]
        totals = _component_totals_for_day(df, d)
        services_total = totals["_services_total"]
        if services_total <= 0:
            continue

        # Find top component
        comps = [(k, v) for k, v in totals.items() if k != "_services_total"]
        top_name, top_cost = max(comps, key=lambda kv: kv[1])
        top_share = float(top_cost / services_total) if services_total > 0 else 0.0

        out.append(
            {
                "type": "top_service_by_cost_single_day",
                "params": {"day": _to_iso(d)},
                "metadata": {
                    "expected_top_service": top_name,
                },
                "targets": [
                    {
                        "key": "top_service_cost",
                        "value": float(top_cost),
                        "tol_abs": tol_abs,
                    },
                    {
                        "key": "top_service_share",
                        "value": float(top_share),
                        "tol_abs": 1e-4,
                    },
                ],
                # Allow date and (optionally) the service label token (not a number).
                "allowed_numbers": [_to_iso(d)],
                "required_tokens": [top_name],
            }
        )
    if len(out) < n:
        raise RuntimeError("Could not sample enough non-zero days for top_service_by_cost.")
    return out


def sample_top_account_by_cost(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    tol_abs: float = 0.01,
) -> list[dict[str, Any]]:
    days = _candidate_days(df)
    out: list[dict[str, Any]] = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        d = days[int(rng.integers(0, len(days)))]
        by_acc = _per_account_day_cost(df, d)
        by_acc = by_acc[by_acc > 0]  # focus on active accounts
        if by_acc.empty:
            continue

        top_acc = int(by_acc.idxmax())
        top_cost = float(by_acc.loc[top_acc])

        out.append(
            {
                "type": "top_account_by_total_cost_single_day",
                "params": {"day": _to_iso(d)},
                "metadata": {"expected_account_id": top_acc},
                "targets": [
                    {
                        "key": "top_account_total_cost",
                        "value": top_cost,
                        "tol_abs": tol_abs,
                    }
                ],
                "allowed_numbers": [_to_iso(d), str(top_acc)],
                "required_tokens": [str(top_acc)],
            }
        )
    if len(out) < n:
        raise RuntimeError("Could not sample enough non-zero days for top_account_by_cost.")
    return out


def sample_p95_account_day_cost(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    min_active_accounts: int = 20,
    tol_abs: float = 0.01,
) -> list[dict[str, Any]]:
    days = _candidate_days(df)
    out: list[dict[str, Any]] = []
    tries = 0
    while len(out) < n and tries < n * 30:
        tries += 1
        d = days[int(rng.integers(0, len(days)))]
        by_acc = _per_account_day_cost(df, d)
        by_acc = by_acc[by_acc > 0]
        if len(by_acc) < min_active_accounts:
            continue

        p95 = float(np.percentile(by_acc.values, 95))
        out.append(
            {
                "type": "p95_account_day_total_cost_single_day",
                "params": {"day": _to_iso(d)},
                "targets": [
                    {
                        "key": "p95_account_day_total_cost",
                        "value": p95,
                        "tol_abs": tol_abs,
                    }
                ],
                # Allow date and the "95" token because the question will mention 95th percentile / p95.
                "allowed_numbers": [_to_iso(d), "95"],
            }
        )
    if len(out) < n:
        raise RuntimeError("Could not sample enough days with sufficient active accounts for p95.")
    return out


def _service_row_total(df: pd.DataFrame) -> pd.Series:
    return df[SERVICE_ONLY_COLS].sum(axis=1)


def _active_service_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    s = _service_row_total(df)
    return df.loc[s > 0].copy()


def _total_cost_services_day(df: pd.DataFrame, day: date) -> float:
    sub = df[df["date"] == day]
    sub = _active_service_rows(sub)
    if sub.empty:
        return 0.0
    return float(_service_row_total(sub).sum())


def _active_service_accounts(df: pd.DataFrame, day: date) -> set[int]:
    sub = df[df["date"] == day]
    sub = _active_service_rows(sub)
    if sub.empty:
        return set()
    return set(sub["account_id"].astype("int64").unique().tolist())


def _per_account_service_costs(df: pd.DataFrame, day: date) -> pd.Series:
    sub = df[df["date"] == day]
    sub = _active_service_rows(sub)
    if sub.empty:
        return pd.Series(dtype=float)
    row_total = _service_row_total(sub)
    per_acc = row_total.groupby(sub["account_id"]).sum()
    per_acc = per_acc[per_acc > 0]
    return per_acc


def sample_total_cost_services_single_day(
    df: pd.DataFrame, n: int, rng: np.random.Generator, tol_abs: float = 0.01
) -> list[dict[str, Any]]:
    days = _sample_values(_candidate_days(df), n, rng)
    out = []
    for d in days:
        val = _total_cost_services_day(df, d)
        out.append(
            {
                "type": "total_cost_services_single_day",
                "params": {"day": _to_iso(d)},
                "targets": [{"key": "total_cost_services", "value": val, "tol_abs": tol_abs}],
                "allowed_numbers": [_to_iso(d)],
            }
        )
    return out


def sample_avg_margin_per_account_day_single_day(
    df: pd.DataFrame, n: int, rng: np.random.Generator, tol_abs: float = 0.01
) -> list[dict[str, Any]]:
    days = _sample_values(_candidate_days(df), n, rng)
    out = []
    for d in days:
        per_acc = _per_account_service_costs(df, d)
        active = int(len(per_acc))
        if active == 0:
            margin = float(PRICE_PER_ACCOUNT_DAY_USD)  # no active -> cost 0
        else:
            avg_cost = float(per_acc.sum() / active)
            margin = float(PRICE_PER_ACCOUNT_DAY_USD) - avg_cost
        out.append(
            {
                "type": "avg_margin_per_active_account_day_single_day",
                "params": {"day": _to_iso(d)},
                "targets": [
                    {
                        "key": "avg_margin_per_account_day",
                        "value": margin,
                        "tol_abs": tol_abs,
                    }
                ],
                "allowed_numbers": [_to_iso(d)],
            }
        )
    return out


def sample_pct_accounts_above_cap_single_day(
    df: pd.DataFrame, n: int, rng: np.random.Generator, tol_abs: float = 1e-4
) -> list[dict[str, Any]]:
    days = _sample_values(_candidate_days(df), n, rng)
    out = []
    for d in days:
        per_acc = _per_account_service_costs(df, d)
        if per_acc.empty:
            pct = 0.0
        else:
            pct = float((per_acc > float(COST_CAP_USD)).mean())
        out.append(
            {
                "type": "pct_accounts_above_cost_cap_single_day",
                "params": {"day": _to_iso(d)},
                "metadata": {"cap_usd": float(COST_CAP_USD)},
                "targets": [{"key": "pct_accounts_above_cap", "value": pct, "tol_abs": tol_abs}],
                "allowed_numbers": [_to_iso(d)],
            }
        )
    return out


def sample_user_churn_single_day(
    df: pd.DataFrame, n: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    # Need day with previous day present
    days = _candidate_days(df)
    day_set = set(days)
    valid = [
        d for d in days if (d.toordinal() - 1) and (date.fromordinal(d.toordinal() - 1) in day_set)
    ]
    if not valid:
        raise RuntimeError("No days with previous-day present; cannot sample churn.")
    picked = _sample_values(valid, n, rng)

    out = []
    for d in picked:
        prev = date.fromordinal(d.toordinal() - 1)
        users_today = _active_service_accounts(df, d)
        users_prev = _active_service_accounts(df, prev)
        new_users = int(len(users_today - users_prev))
        churned = int(len(users_prev - users_today))
        net = int(len(users_today) - len(users_prev))
        out.append(
            {
                "type": "user_churn_single_day_vs_prev_day",
                "params": {"day": _to_iso(d), "prev_day": _to_iso(prev)},
                "targets": [
                    {"key": "new_users", "value": new_users, "tol_abs": 0.0},
                    {"key": "churned_users", "value": churned, "tol_abs": 0.0},
                    {"key": "net_user_change", "value": net, "tol_abs": 0.0},
                ],
                "allowed_numbers": [_to_iso(d), _to_iso(prev)],
            }
        )
    return out


def sample_max_total_cost_services_day_in_window(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    window_min_days: int = 3,
    window_max_days: int = 14,
    tol_abs: float = 0.01,
) -> list[dict[str, Any]]:
    days = _candidate_days(df)
    if len(days) < window_min_days:
        raise ValueError("Not enough unique days to sample windows.")

    out = []
    for _ in range(n):
        w = int(rng.integers(window_min_days, window_max_days + 1))
        start_idx = int(rng.integers(0, max(1, len(days) - w)))
        start = days[start_idx]
        end = days[min(start_idx + w - 1, len(days) - 1)]

        # compute daily service totals inside window
        window_days = [d for d in days if start <= d <= end]
        daily = [(d, _total_cost_services_day(df, d)) for d in window_days]
        # pick max; tie-breaker earliest day for determinism
        max_day, max_val = sorted(daily, key=lambda x: (-x[1], x[0]))[0]

        out.append(
            {
                "type": "max_total_cost_services_day_in_window",
                "params": {"start": _to_iso(start), "end": _to_iso(end)},
                "metadata": {"max_day": _to_iso(max_day)},
                "targets": [
                    {
                        "key": "max_total_cost_services",
                        "value": float(max_val),
                        "tol_abs": tol_abs,
                    },
                ],
                # We include max_day so the question generator is allowed to mention it if needed
                # (but it doesn’t have to — you can keep it implicit).
                "allowed_numbers": [_to_iso(start), _to_iso(end), _to_iso(max_day)],
            }
        )
    return out


# ------------------------- Driver -------------------------


FACT_SAMPLERS = {
    "total_cost_single_day": sample_total_cost_single_day,
    "total_cost_window": sample_total_cost_window,
    "top_service_by_cost_single_day": sample_top_service_by_cost,
    "top_account_by_total_cost_single_day": sample_top_account_by_cost,
    "p95_account_day_total_cost_single_day": sample_p95_account_day_cost,
    "total_cost_services_single_day": sample_total_cost_services_single_day,
    "avg_margin_per_active_account_day_single_day": sample_avg_margin_per_account_day_single_day,
    "pct_accounts_above_cost_cap_single_day": sample_pct_accounts_above_cap_single_day,
    "user_churn_single_day_vs_prev_day": sample_user_churn_single_day,
    "max_total_cost_services_day_in_window": sample_max_total_cost_services_day_in_window,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate fact instances (oracle targets) from CSV.")
    ap.add_argument("--csv", required=True, help="Path to joint_costs_daily CSV")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=50, help="Samples per question type")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument(
        "--types",
        nargs="*",
        default=list(FACT_SAMPLERS.keys()),
        help=f"Subset of types to generate. Options: {sorted(FACT_SAMPLERS.keys())}",
    )
    args = ap.parse_args()

    df = load_csv(args.csv)
    rng = np.random.default_rng(args.seed)
    anchor_day = _anchor_day_from_df(df)

    records: list[dict[str, Any]] = []
    next_id = 1

    for t in args.types:
        if t not in FACT_SAMPLERS:
            raise ValueError(f"Unknown type: {t}. Allowed: {sorted(FACT_SAMPLERS.keys())}")
        sampler = FACT_SAMPLERS[t]
        samples = sampler(df=df, n=args.n, rng=rng)

        for s in samples:
            s["id"] = next_id
            next_id += 1
            s["anchor_day"] = _to_iso(anchor_day)
            s["csv_fingerprint"] = {
                # minimal reproducibility hints
                "csv_path": str(Path(args.csv)),
                "anchor_day": _to_iso(anchor_day),
                "seed": args.seed,
            }
            records.append(s)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} fact instances to {out_path}")


if __name__ == "__main__":
    main()
