from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

"""
Account×day anomaly detector aligned to the synthetic anomaly injector.

Key design goals (vs previous detector):
- Score and gate on *raw* costs / volumes / chats (matches injection mechanisms).
- Robust 14-day per-account baseline (median + MAD on active/non-zero days).
- NO peer fallback by default (peer fallback + tiny scale floors caused false positives).
- Materiality gates (ratio + absolute excess) to suppress small-noise spikes.
- Drivers align with label types: {cost_spike, volume_spike, cap_spike}.

Public API is kept compatible with the previous detector:
- detect_anomaly_for_end_day(...)
- format_detector_brief(...)
"""

# --------------------------------------------------------------------------------------
# Columns (match your synthetic generator / injector)
# --------------------------------------------------------------------------------------

# "Services" costs are stored as derived totals in the dataset.
SERVICE_COST_COLS: tuple[str, ...] = (
    "total_cost_tasks",
    "total_cost_classifications",
    "cost_amocrm_call",
)
DIALOG_COST_COL = "cost_dialog"

# Optional usage counters (expected to exist in your synthetic data)
VOLUME_COLS: tuple[str, ...] = ("n_tasks", "n_classifications", "n_amocrm_calls")

# Reasonable per-metric scale floors for robust stats on *log1p* metrics.
# (Prevent division by ~0 when baseline is near-constant.)
LOG_SCALE_FLOOR: dict[str, float] = {
    "log_services_cost": 0.20,
    "log_dialog_cost": 0.20,
    "log_total_cost": 0.20,
    "log_n_tasks": 0.20,
    "log_n_classifications": 0.20,
    "log_n_amocrm_calls": 0.20,
    "log_n_chats": 0.20,
}

# Absolute gates (used mainly when baseline is sparse/untrusted).
ABS_MIN_COUNTS: dict[str, float] = {
    # tuned for your synthetic injections where multipliers are huge (e.g., x50–x100)
    "n_tasks": 200.0,
    "n_classifications": 300.0,
    "n_amocrm_calls": 100.0,
}
ABS_MIN_COST_NO_HISTORY = 200.0  # if an account has no baseline, only flag very large cost

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def format_detector_brief(det: dict[str, Any]) -> str:
    d = det.get("decision", {})
    day = d.get("day", "?")
    if not d:
        return "No detector output."
    if not d.get("anomaly", False):
        return f"No anomaly detected on {day}."
    return (
        f"Anomaly detected on {day} "
        f"(driver={d.get('driver')}, flagged_accounts={d.get('flagged_accounts')}, "
        f"excess_cost={float(d.get('excess_cost', 0.0)):.2f}, "
        f"severity={float(d.get('severity', 0.0)):.2f})."
    )


def _parse_iso_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start: date, end: date) -> list[date]:
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date

    # numeric columns (fill missing with 0)
    numeric_cols = [*SERVICE_COST_COLS, DIALOG_COST_COL, *VOLUME_COLS]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0

    # ids
    out["account_id"] = pd.to_numeric(out["account_id"], errors="coerce")
    out["chat_id"] = pd.to_numeric(out["chat_id"], errors="coerce")

    return out


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = den.replace(0, np.nan)
    return (num / den2).fillna(0.0)


def _robust_center_scale(values: np.ndarray, scale_floor: float) -> tuple[float, float]:
    """
    Robust center/scale using median and MAD (scaled by 1.4826).
    Falls back to a reasonable floor to avoid absurd z-scores when MAD==0.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, scale_floor

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < scale_floor:
        # IQR fallback if MAD collapses
        q75, q25 = np.percentile(values, [75, 25])
        iqr = float(q75 - q25)
        scale = max(scale_floor, iqr / 1.349 if iqr > 0 else scale_floor)

    return med, scale


def _robust_z(x: np.ndarray, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.asarray(med, dtype=float)
    scale = np.asarray(scale, dtype=float)
    scale = np.where(~np.isfinite(scale) | (scale <= 0), 1.0, scale)
    out = (x - med) / scale
    out[~np.isfinite(out)] = 0.0
    return out


def _account_day_features(df: pd.DataFrame, days: Sequence[date]) -> pd.DataFrame:
    """
    Build account×day features for the requested days.
    """
    df2 = df[df["date"].isin(days)].copy()
    if df2.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "account_id",
                "n_chats",
                "n_tasks",
                "n_classifications",
                "n_amocrm_calls",
                "services_cost",
                "dialog_cost",
                "total_cost",
                "tasks_per_chat",
                "dialog_cost_per_chat",
                # log versions for scoring
                "log_services_cost",
                "log_dialog_cost",
                "log_total_cost",
                "log_n_tasks",
                "log_n_classifications",
                "log_n_amocrm_calls",
                "log_n_chats",
            ]
        )

    df2["services_cost"] = df2[list(SERVICE_COST_COLS)].sum(axis=1)
    df2["dialog_cost"] = df2[DIALOG_COST_COL]

    agg = (
        df2.groupby(["date", "account_id"], as_index=False)
        .agg(
            n_chats=("chat_id", "nunique"),
            services_cost=("services_cost", "sum"),
            dialog_cost=("dialog_cost", "sum"),
            n_tasks=("n_tasks", "sum"),
            n_classifications=("n_classifications", "sum"),
            n_amocrm_calls=("n_amocrm_calls", "sum"),
        )
        .sort_values(["date", "account_id"])
    )

    agg["total_cost"] = agg["services_cost"] + agg["dialog_cost"]
    agg["tasks_per_chat"] = _safe_div(agg["n_tasks"], agg["n_chats"])
    agg["dialog_cost_per_chat"] = _safe_div(agg["dialog_cost"], agg["n_chats"])

    # log versions (for robust z scoring)
    agg["log_services_cost"] = np.log1p(agg["services_cost"].astype(float))
    agg["log_dialog_cost"] = np.log1p(agg["dialog_cost"].astype(float))
    agg["log_total_cost"] = np.log1p(agg["total_cost"].astype(float))
    agg["log_n_tasks"] = np.log1p(agg["n_tasks"].astype(float))
    agg["log_n_classifications"] = np.log1p(agg["n_classifications"].astype(float))
    agg["log_n_amocrm_calls"] = np.log1p(agg["n_amocrm_calls"].astype(float))
    agg["log_n_chats"] = np.log1p(agg["n_chats"].astype(float))

    return agg


def _day_context_table(df: pd.DataFrame, days: Sequence[date]) -> list[dict[str, Any]]:
    """
    Day-level context only (not used for decision).
    """
    df2 = df[df["date"].isin(days)].copy()
    if df2.empty:
        return []

    df2["services_cost"] = df2[list(SERVICE_COST_COLS)].sum(axis=1)

    by_day = (
        df2.groupby("date", as_index=False)
        .agg(
            services_total_cost=("services_cost", "sum"),
            dialog_total_cost=(DIALOG_COST_COL, "sum"),
            active_accounts=("account_id", "nunique"),
            active_chats=("chat_id", "nunique"),
            sum_n_tasks=("n_tasks", "sum"),
            sum_n_classifications=("n_classifications", "sum"),
            sum_n_amocrm_calls=("n_amocrm_calls", "sum"),
        )
        .sort_values("date")
    )
    return by_day.to_dict("records")


def _compute_account_baseline_stats(
    baseline_ad: pd.DataFrame, metrics: Sequence[str]
) -> pd.DataFrame:
    """
    For each account_id, compute baseline center/scale per metric on non-zero values.
    Stores:
      {metric}__med, {metric}__scale, {metric}__points
    """
    if baseline_ad.empty:
        return pd.DataFrame()

    def _one_account(g: pd.DataFrame) -> pd.Series:
        out: dict[str, float] = {"baseline_rows": float(len(g))}
        for m in metrics:
            vals = g[m].astype(float)
            nz = vals[vals > 0.0]  # active / meaningful values only
            out[f"{m}__points"] = float(len(nz))
            if len(nz) == 0:
                out[f"{m}__med"] = 0.0
                # floor by metric if log metric
                floor = LOG_SCALE_FLOOR.get(m, 1.0)
                out[f"{m}__scale"] = float(floor)
                continue
            floor = LOG_SCALE_FLOOR.get(m, 1.0)
            med, sc = _robust_center_scale(nz.values, scale_floor=float(floor))
            out[f"{m}__med"] = float(med)
            out[f"{m}__scale"] = float(sc)
        return pd.Series(out)

    stats = baseline_ad.groupby("account_id", sort=False).apply(_one_account)
    stats.index = stats.index.astype("int64")
    return stats


@dataclass(frozen=True)
class Thresholds:
    z_thr: float
    chat_cap: int | None
    min_delta: float
    min_accounts: int
    min_baseline_points: int


def detect_anomaly_for_end_day(
    *,
    csv_path: str,
    end_day: str | date,
    baseline_days: int = 14,
    context_days: int = 7,
    gap_days: int = 2,
    # thresholds to calibrate:
    z_thr: float = 4.5,
    # legacy alias (old detector used z_threshold):
    z_threshold: float | None = None,
    # IMPORTANT default: use HARD cap for concurrency anomaly by default
    chat_cap: int | None = 80,
    min_delta: float = 0.0,
    min_accounts: int = 1,
    min_baseline_points: int = 7,
    min_account_impact_services: float = 10.0,
    min_account_impact_dialog: float = 1.0,
    top_k_accounts: int = 10,
    debug: bool = False,
    # New (safe) knobs — optional; old callers can ignore.
    ratio_cost: float = 5.0,
    ratio_volume: float = 5.0,
    abs_chat_jump: int = 10,
    enable_chat_jump: bool = False,
    # Debug / evaluation: include per-account decisions in output JSON
    include_account_decisions: bool = False,
    # accept extra args from old CLI (mix thresholds, cost cap, etc.)
    **_unused: Any,
) -> dict[str, Any]:
    """
    Detect anomalies on a given end_day using a 14-day per-account robust baseline.

    Drivers emitted (aligned with labels.jsonl):
      - cost_spike
      - volume_spike
      - cap_spike

    Day-level anomaly is True if:
      flagged_accounts >= min_accounts AND excess_cost >= min_delta
    """
    end = _parse_iso_date(end_day) if isinstance(end_day, str) else end_day
    if z_threshold is not None:
        z_thr = float(z_threshold)

    thresholds = Thresholds(
        z_thr=float(z_thr),
        chat_cap=chat_cap,
        min_delta=float(min_delta),
        min_accounts=int(min_accounts),
        min_baseline_points=int(min_baseline_points),
    )

    logger = logging.getLogger(__name__)
    df = _ensure_types(pd.read_csv(csv_path))

    # windows
    # Baseline ends *before* the last `gap_days` to avoid contaminating the baseline
    # when anomalies last multiple days.
    if gap_days < 0:
        gap_days = 0
    baseline_end = end - timedelta(days=gap_days + 1)  # inclusive
    baseline_start = baseline_end - timedelta(days=baseline_days - 1)
    baseline_range = _daterange(baseline_start, baseline_end)

    context_start = end - timedelta(days=context_days - 1)
    context_range = _daterange(context_start, end)

    end_ad = _account_day_features(df, [end])
    baseline_ad = _account_day_features(df, baseline_range)

    if end_ad.empty:
        return {
            "params": {
                "baseline_days": baseline_days,
                "context_days": context_days,
                "gap_days": int(gap_days),
                "enable_chat_jump": bool(enable_chat_jump),
                "z_thr": thresholds.z_thr,
                "chat_cap": thresholds.chat_cap,
                "min_delta": thresholds.min_delta,
                "min_accounts": thresholds.min_accounts,
                "min_baseline_points": thresholds.min_baseline_points,
                "top_k_accounts": int(top_k_accounts),
                "include_account_decisions": bool(include_account_decisions),
            },
            "decision": {
                "day": end.isoformat(),
                "anomaly": False,
                "driver": "none",
                "flagged_accounts": 0,
                "excess_cost": 0.0,
                "severity": 0.0,
                "why": f"No data for end_day={end.isoformat()}",
                "by_type": [],
                "top_accounts": [],
            },
            "context_last_days": _day_context_table(df, context_range),
        }

    # Scoring metrics (log1p)
    score_metrics = [
        "log_services_cost",
        "log_dialog_cost",
        "log_total_cost",
        "log_n_tasks",
        "log_n_classifications",
        "log_n_amocrm_calls",
        "log_n_chats",
    ]

    acc_stats = _compute_account_baseline_stats(
        baseline_ad,
        metrics=score_metrics,
    ).rename(columns={"baseline_rows": "log_baseline_rows"})

    # Raw baselines for excess/ratio gates (median on non-zero values)
    raw_metrics = [
        "services_cost",
        "dialog_cost",
        "total_cost",
        "n_tasks",
        "n_classifications",
        "n_amocrm_calls",
        "n_chats",
    ]
    raw_stats = _compute_account_baseline_stats(
        baseline_ad.assign(
            services_cost=baseline_ad["services_cost"],
            dialog_cost=baseline_ad["dialog_cost"],
            total_cost=baseline_ad["total_cost"],
            n_tasks=baseline_ad["n_tasks"],
            n_classifications=baseline_ad["n_classifications"],
            n_amocrm_calls=baseline_ad["n_amocrm_calls"],
            n_chats=baseline_ad["n_chats"],
        ),
        metrics=raw_metrics,
    ).rename(columns={"baseline_rows": "raw_baseline_rows"})

    # merge onto end-day accounts
    end_scored = end_ad.merge(acc_stats.reset_index(), on="account_id", how="left")
    end_scored = end_scored.merge(raw_stats.reset_index(), on="account_id", how="left")

    # robust z vs own baseline only (no peer fallback)
    for m in score_metrics:
        med_col = f"{m}__med"
        sc_col = f"{m}__scale"
        pts_col = f"{m}__points"
        pts = (
            end_scored.get(pts_col, pd.Series(0.0, index=end_scored.index))
            .fillna(0.0)
            .astype(float)
        )
        has_baseline = pts >= thresholds.min_baseline_points

        z = np.zeros(len(end_scored), dtype=float)
        if has_baseline.any():
            z[has_baseline.values] = _robust_z(
                end_scored.loc[has_baseline, m].astype(float).values,
                end_scored.loc[has_baseline, med_col].astype(float).values,
                end_scored.loc[has_baseline, sc_col].astype(float).values,
            )

        end_scored[f"z__{m}"] = z
        end_scored[f"z_pos__{m}"] = np.maximum(z, 0.0)
        end_scored[f"z_sev__{m}"] = np.abs(z)

    # Excess costs for impact accounting (vs baseline median on active days).
    def _excess(today: pd.Series, med: pd.Series, pts: pd.Series) -> pd.Series:
        med2 = med.fillna(0.0)
        pts2 = pts.fillna(0.0)
        raw = (today - med2).clip(lower=0.0)
        # if no baseline points -> treat excess as 0 to avoid "first activation" spikes
        return pd.Series(np.where(pts2 > 0.0, raw, 0.0), index=today.index)

    end_scored["excess_services_cost"] = _excess(
        end_scored["services_cost"].astype(float),
        end_scored.get("services_cost__med", pd.Series(np.nan, index=end_scored.index)),
        end_scored.get("services_cost__points", pd.Series(0.0, index=end_scored.index)),
    )
    end_scored["excess_dialog_cost"] = _excess(
        end_scored["dialog_cost"].astype(float),
        end_scored.get("dialog_cost__med", pd.Series(np.nan, index=end_scored.index)),
        end_scored.get("dialog_cost__points", pd.Series(0.0, index=end_scored.index)),
    )
    end_scored["excess_total_cost"] = (
        end_scored["excess_services_cost"] + end_scored["excess_dialog_cost"]
    )

    # Helpful baselines (raw) for ratio gates
    def _baseline_med(col: str) -> pd.Series:
        return (
            end_scored.get(f"{col}__med", pd.Series(0.0, index=end_scored.index))
            .fillna(0.0)
            .astype(float)
        )

    def _baseline_pts(col: str) -> pd.Series:
        return (
            end_scored.get(f"{col}__points", pd.Series(0.0, index=end_scored.index))
            .fillna(0.0)
            .astype(float)
        )

    # -------------------- account-level triggers (aligned with injected types) --------------------

    # COST SPIKE: services and/or dialog costs jump materially vs baseline
    med_services = _baseline_med("services_cost")
    pts_services = _baseline_pts("services_cost")
    med_dialog = _baseline_med("dialog_cost")
    pts_dialog = _baseline_pts("dialog_cost")

    has_cost_baseline = (pts_services >= thresholds.min_baseline_points) | (
        pts_dialog >= thresholds.min_baseline_points
    )

    services_ratio = (end_scored["services_cost"].astype(float) + 1.0) / (med_services + 1.0)
    dialog_ratio = (end_scored["dialog_cost"].astype(float) + 1.0) / (med_dialog + 1.0)

    # robust z on log_total_cost / log_services_cost provides extra stability; require z>=z_thr as well
    z_cost = np.maximum(end_scored["z_pos__log_total_cost"], end_scored["z_pos__log_services_cost"])

    # Store key ratios/gates for debugging and evaluation
    end_scored["ratio__services_cost"] = services_ratio
    end_scored["ratio__dialog_cost"] = dialog_ratio
    end_scored["cost_has_baseline"] = has_cost_baseline
    end_scored["cost_pass_z"] = z_cost >= thresholds.z_thr
    end_scored["cost_pass_ratio"] = (services_ratio >= float(ratio_cost)) | (
        dialog_ratio >= float(ratio_cost)
    )
    end_scored["cost_pass_impact"] = (
        end_scored["excess_services_cost"] >= float(min_account_impact_services)
    ) | (end_scored["excess_dialog_cost"] >= float(min_account_impact_dialog))

    strong_ratio_cost = 8.0
    strong_excess_services = 50.0

    cost_pass_strong = (end_scored["ratio__services_cost"] >= strong_ratio_cost) & (
        end_scored["excess_services_cost"] >= strong_excess_services
    )

    end_scored["flag_cost_spike"] = (
        has_cost_baseline
        & (z_cost >= thresholds.z_thr)
        & ((services_ratio >= float(ratio_cost)) | (dialog_ratio >= float(ratio_cost)))
        & (
            (end_scored["excess_services_cost"] >= float(min_account_impact_services))
            | (end_scored["excess_dialog_cost"] >= float(min_account_impact_dialog))
        )
    )
    # override for strong ratio cases
    end_scored.loc[cost_pass_strong, "flag_cost_spike"] = True
    end_scored.loc[cost_pass_strong, "cost_reason"] = "strong_ratio_override"
    # Sparse-history fallback for cost spike (VERY conservative)
    end_scored.loc[~has_cost_baseline, "flag_cost_spike"] = (
        end_scored.loc[~has_cost_baseline, "total_cost"].astype(float)
        >= float(ABS_MIN_COST_NO_HISTORY)
    ) & (
        (
            end_scored.loc[~has_cost_baseline, "services_cost"].astype(float)
            >= float(min_account_impact_services)
        )
        | (
            end_scored.loc[~has_cost_baseline, "dialog_cost"].astype(float)
            >= float(min_account_impact_dialog)
        )
    )

    # Debug: capture the conservative no-baseline fallback condition explicitly
    end_scored["cost_pass_fallback"] = (~has_cost_baseline) & (
        (end_scored["total_cost"].astype(float) >= float(ABS_MIN_COST_NO_HISTORY))
        & (
            (end_scored["services_cost"].astype(float) >= float(min_account_impact_services))
            | (end_scored["dialog_cost"].astype(float) >= float(min_account_impact_dialog))
        )
    )

    # Human-readable reason for why cost_spike did / did not trigger (useful for FN analysis)
    def _cost_reason_row(r: pd.Series) -> str:
        if bool(r.get("flag_cost_spike", False)):
            return "flag_cost_spike"
        if bool(r.get("cost_has_baseline", False)):
            if not bool(r.get("cost_pass_z", False)):
                return "z_below_thr"
            if not bool(r.get("cost_pass_ratio", False)):
                return "ratio_below_thr"
            if not bool(r.get("cost_pass_impact", False)):
                return "impact_below_thr"
            return "not_flagged_other"
        else:
            if bool(r.get("cost_pass_fallback", False)):
                return "fallback_abs_trigger"
            return "no_baseline_or_small_cost"

    end_scored["cost_reason"] = end_scored.apply(_cost_reason_row, axis=1)

    # VOLUME SPIKE: big jump in counts (and typically associated services_cost impact)
    # Use counts if baseline exists; otherwise require very large absolute counts + cost impact.
    vol_cols = [
        ("n_tasks", "log_n_tasks"),
        ("n_classifications", "log_n_classifications"),
        ("n_amocrm_calls", "log_n_amocrm_calls"),
    ]

    vol_flags = []
    vol_sevs = []
    for raw_col, log_col in vol_cols:
        med = _baseline_med(raw_col)
        pts = _baseline_pts(raw_col)
        has_base = pts >= thresholds.min_baseline_points

        ratio = (end_scored[raw_col].astype(float) + 1.0) / (med + 1.0)
        z = end_scored[f"z_pos__{log_col}"]

        # baseline-based trigger
        f_base = has_base & (z >= thresholds.z_thr) & (ratio >= float(ratio_volume))

        # sparse fallback (absolute gate)
        abs_min = float(ABS_MIN_COUNTS.get(raw_col, 25.0))
        f_abs = (~has_base) & (end_scored[raw_col].astype(float) >= abs_min)

        vol_flags.append(f_base | f_abs)
        sev = np.where(f_abs, np.log1p(end_scored[raw_col].astype(float)), z)
        vol_sevs.append(sev)

        end_scored[f"ratio__{raw_col}"] = ratio

    vol_any = vol_flags[0] | vol_flags[1] | vol_flags[2]
    vol_sev = np.maximum.reduce([np.asarray(s) for s in vol_sevs])

    # Strong absolute volume spike (matches your injected pattern: ~303+)
    vol_abs_strong = (
        (end_scored["n_tasks"].astype(float) >= float(ABS_MIN_COUNTS["n_tasks"]))
        | (
            end_scored["n_classifications"].astype(float)
            >= float(ABS_MIN_COUNTS["n_classifications"])
        )
        | (end_scored["n_amocrm_calls"].astype(float) >= float(ABS_MIN_COUNTS["n_amocrm_calls"]))
    )

    # Flag volume spike if:
    # - volume trigger fired AND
    #   - either it is a "strong absolute" volume spike (even if cost is 0),
    #   - or it has meaningful services-cost impact.
    end_scored["flag_volume_spike"] = vol_any & (
        vol_abs_strong
        | (
            (end_scored["services_cost"].astype(float) > 0.0)
            & (end_scored["excess_services_cost"] >= float(min_account_impact_services))
        )
    )
    end_scored["vol_severity"] = vol_sev

    # CAP SPIKE: hard cap violation OR large jump in chats vs baseline
    cap_flag = False
    if thresholds.chat_cap is not None:
        cap_flag = end_scored["n_chats"].astype(int) >= int(thresholds.chat_cap)

    med_chats = _baseline_med("n_chats")
    pts_chats = _baseline_pts("n_chats")
    has_chat_baseline = pts_chats >= thresholds.min_baseline_points

    # require a meaningful absolute jump if relying on baseline rather than hard cap
    z_chats = end_scored["z_pos__log_n_chats"]
    jump = end_scored["n_chats"].astype(float) - med_chats
    end_scored["flag_cap_spike"] = cap_flag
    if enable_chat_jump:
        end_scored["flag_cap_spike"] = cap_flag | (
            has_chat_baseline
            & (z_chats >= thresholds.z_thr)
            & (jump >= float(abs_chat_jump))
            & ((end_scored["n_chats"].astype(float) + 1.0) / (med_chats + 1.0) >= 2.0)
        )

    # -------------------- assign per-account driver --------------------
    def _account_driver(row: pd.Series) -> tuple[str, float, float]:
        # Force cap_spike to win when present (label alignment).
        if bool(row["flag_cap_spike"]):
            sev = (
                float(row["n_chats"] - int(thresholds.chat_cap) + 1)
                if thresholds.chat_cap is not None
                and int(row["n_chats"]) >= int(thresholds.chat_cap)
                else float(row.get("z_pos__log_n_chats", 0.0))
            )
            return ("cap_spike", sev, float(row.get("excess_total_cost", 0.0)))

        if bool(row["flag_volume_spike"]):
            return (
                "volume_spike",
                float(row.get("vol_severity", 0.0)),
                float(row.get("excess_services_cost", 0.0)),
            )

        if bool(row["flag_cost_spike"]):
            sev = float(
                max(
                    row.get("z_pos__log_total_cost", 0.0),
                    row.get("z_pos__log_services_cost", 0.0),
                    row.get("z_pos__log_dialog_cost", 0.0),
                )
            )
            return ("cost_spike", sev, float(row.get("excess_total_cost", 0.0)))

        return ("none", 0.0, 0.0)

    tmp = end_scored.apply(_account_driver, axis=1, result_type="expand")
    tmp.columns = ["account_driver", "account_severity", "account_impact_cost"]
    end_scored["account_driver"] = tmp["account_driver"]
    end_scored["account_severity"] = tmp["account_severity"].astype(float)
    end_scored["account_impact_cost"] = tmp["account_impact_cost"].astype(float)

    flagged = end_scored[end_scored["account_driver"] != "none"].copy()

    by_type = (
        flagged.groupby("account_driver", dropna=False)
        .agg(
            flagged_accounts=("account_id", "nunique"),
            total_impact_cost=("account_impact_cost", "sum"),
            max_severity=("account_severity", "max"),
        )
        .reset_index()
        .sort_values(["total_impact_cost", "max_severity"], ascending=False)
    )

    total_flagged_accounts = int(flagged["account_id"].nunique()) if not flagged.empty else 0
    total_excess_cost = float(flagged["account_impact_cost"].sum()) if not flagged.empty else 0.0
    day_severity = float(flagged["account_severity"].max()) if not flagged.empty else 0.0

    day_is_anomaly = (total_flagged_accounts >= thresholds.min_accounts) and (
        total_excess_cost >= thresholds.min_delta
    )
    # If there are cap spikes with 0 excess cost and min_delta>0, day may not be flagged.
    # That behaviour is intentional given min_delta semantics.

    day_driver = str(by_type.iloc[0]["account_driver"]) if not by_type.empty else "none"

    # Evidence: top accounts by impact cost, then severity
    evidence_cols = [
        "account_id",
        "account_driver",
        "account_severity",
        "account_impact_cost",
        "n_chats",
        "n_tasks",
        "n_classifications",
        "n_amocrm_calls",
        "services_cost",
        "dialog_cost",
        "total_cost",
        "excess_services_cost",
        "excess_dialog_cost",
        "excess_total_cost",
        # ratios
        "ratio__n_tasks",
        "ratio__n_classifications",
        "ratio__n_amocrm_calls",
        # z scores
        "z_pos__log_services_cost",
        "z_pos__log_dialog_cost",
        "z_pos__log_total_cost",
        "z_pos__log_n_tasks",
        "z_pos__log_n_classifications",
        "z_pos__log_n_amocrm_calls",
        "z_pos__log_n_chats",
        # baselines (raw)
        "services_cost__med",
        "dialog_cost__med",
        "n_tasks__med",
        "n_classifications__med",
        "n_amocrm_calls__med",
        "n_chats__med",
        "services_cost__points",
        "dialog_cost__points",
        "n_tasks__points",
        "n_classifications__points",
        "n_amocrm_calls__points",
        "n_chats__points",
    ]
    for c in evidence_cols:
        if c not in end_scored.columns:
            end_scored[c] = np.nan

    top_accounts = (
        flagged.sort_values(["account_impact_cost", "account_severity"], ascending=False)
        .head(int(top_k_accounts))
        .loc[:, evidence_cols]
        .to_dict("records")
    )

    # For evaluation: include the full set of flagged account ids (not just top_k evidence)
    flagged_account_ids = (
        sorted(int(x) for x in flagged["account_id"].dropna().unique()) if not flagged.empty else []
    )

    # Optional: include per-account decisions for the whole day (useful for FN/FP analysis)
    account_decisions: list[dict[str, Any]] = []
    if include_account_decisions:
        decision_cols = [
            "account_id",
            "account_driver",
            "account_severity",
            "account_impact_cost",
            # raw values
            "services_cost",
            "dialog_cost",
            "total_cost",
            "n_chats",
            "n_tasks",
            "n_classifications",
            "n_amocrm_calls",
            # cost-spike gates
            "flag_cost_spike",
            "cost_reason",
            "cost_has_baseline",
            "cost_pass_z",
            "cost_pass_ratio",
            "cost_pass_impact",
            "cost_pass_fallback",
            "services_cost__med",
            "dialog_cost__med",
            "services_cost__points",
            "dialog_cost__points",
            "ratio__services_cost",
            "ratio__dialog_cost",
            "excess_services_cost",
            "excess_dialog_cost",
            "excess_total_cost",
            "z_pos__log_services_cost",
            "z_pos__log_dialog_cost",
            "z_pos__log_total_cost",
            # volume/cap summaries (helpful when diagnosing "wrong driver")
            "flag_volume_spike",
            "vol_severity",
            "ratio__n_tasks",
            "ratio__n_classifications",
            "ratio__n_amocrm_calls",
            "flag_cap_spike",
            "z_pos__log_n_chats",
        ]
        for c in decision_cols:
            if c not in end_scored.columns:
                end_scored[c] = np.nan
        account_decisions = (
            end_scored.sort_values("account_id").loc[:, decision_cols].to_dict("records")
        )

    why = (
        f"{total_flagged_accounts} account(s) flagged on {end.isoformat()} "
        f"(driver={day_driver}, excess_cost={total_excess_cost:.2f}). "
        f"Detector uses per-account 14-day baseline (median+MAD) with ratio+impact gates; "
        f"peer fallback is disabled to reduce false positives."
    )

    if debug:
        logger.info(
            "end_day=%s | accounts=%d | flagged_accounts=%d | excess_cost=%.2f | driver=%s",
            end.isoformat(),
            int(len(end_scored)),
            int(total_flagged_accounts),
            float(total_excess_cost),
            day_driver,
        )
        if not by_type.empty:
            logger.info("by_type: %s", by_type.to_dict("records"))
        if top_accounts:
            logger.debug("top_accounts: %s", top_accounts[: min(5, len(top_accounts))])

    return {
        "params": {
            "baseline_days": baseline_days,
            "context_days": context_days,
            "gap_days": int(gap_days),
            "enable_chat_jump": bool(enable_chat_jump),
            "z_thr": thresholds.z_thr,
            "chat_cap": thresholds.chat_cap,
            "min_delta": thresholds.min_delta,
            "min_accounts": thresholds.min_accounts,
            "min_baseline_points": thresholds.min_baseline_points,
            "top_k_accounts": int(top_k_accounts),
            "include_account_decisions": bool(include_account_decisions),
            # extra knobs (safe to ignore)
            "ratio_cost": float(ratio_cost),
            "ratio_volume": float(ratio_volume),
            "abs_chat_jump": int(abs_chat_jump),
        },
        "decision": {
            "day": end.isoformat(),
            "anomaly": bool(day_is_anomaly),
            "driver": day_driver,
            "flagged_accounts": total_flagged_accounts,
            "excess_cost": total_excess_cost,
            "severity": day_severity,
            "why": why,
            "by_type": by_type.to_dict("records"),
            "top_accounts": top_accounts,
            "flagged_account_ids": flagged_account_ids,
            "account_decisions": account_decisions,
        },
        "context_last_days": _day_context_table(df, context_range),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run account-day anomaly detector for a specific day."
    )
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV.")
    parser.add_argument("--day", required=True, help="End day in YYYY-MM-DD.")
    parser.add_argument("--baseline-days", type=int, default=14)
    parser.add_argument("--context-days", type=int, default=7)
    parser.add_argument("--z-thr", type=float, default=5.5)
    parser.add_argument("--chat-cap", type=int, default=70)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-accounts", type=int, default=1)
    parser.add_argument("--min-baseline-points", type=int, default=7)
    parser.add_argument("--min-impact-services", type=float, default=20.0)
    parser.add_argument("--min-impact-dialog", type=float, default=1.0)
    parser.add_argument("--ratio-cost", type=float, default=10.0)
    parser.add_argument("--ratio-volume", type=float, default=5.0)
    parser.add_argument("--abs-chat-jump", type=int, default=10)
    parser.add_argument("--top-k-accounts", type=int, default=10)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    out = detect_anomaly_for_end_day(
        csv_path=args.csv,
        end_day=args.day,
        baseline_days=args.baseline_days,
        context_days=args.context_days,
        z_thr=args.z_thr,
        chat_cap=args.chat_cap,
        min_delta=args.min_delta,
        min_accounts=args.min_accounts,
        min_baseline_points=args.min_baseline_points,
        min_account_impact_services=args.min_impact_services,
        min_account_impact_dialog=args.min_impact_dialog,
        ratio_cost=args.ratio_cost,
        ratio_volume=args.ratio_volume,
        abs_chat_jump=args.abs_chat_jump,
        top_k_accounts=args.top_k_accounts,
        debug=args.debug,
    )
    print(json.dumps(out, indent=2, default=str))
