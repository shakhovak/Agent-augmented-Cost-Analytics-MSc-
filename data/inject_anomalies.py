"""Inject configurable anomalies into synthetic cost data and emit JSONL labels.

Supports:
- cost_spike: multiply cost components on the account-day summary row
- volume_spike: scale counts (and roughly scale related costs)
- cap_spike: add rows above max_concurrent and optionally increase costs

Config is read from a YAML file.

NEW (for more reliable experiments / evaluation)
- Labels include per-day effective accounts: `effective_accounts_by_day`.
  This fixes the common issue where an account is "selected" for a cost spike but
  has zero in the targeted component on that day (multiplying 0 keeps it 0).
- Optional per-day sampling: `account_selection.per_day: true`.
  If enabled, we sample from *eligible* accounts on each injected date.
- Window can be an explicit date list: `window.dates: ['2025-10-15', ...]`.

Backward compatible:
- If you don't use `per_day` or `window.dates`, behavior matches the old script,
  but labels are now accurate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required for anomaly injection. Install it (pip install pyyaml)."
    ) from e

from cost_agent_mvp.data.schema import load_csv

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):  # type: ignore
        return x


# Cost columns used when applying "all" components (cost_spike or cap_spike cost bump).
COST_COLUMNS = [
    "cost_dialog",
    "cost_task",
    "cost_classification",
    "cost_qc",
    "cost_check_list",
    "cost_amocrm_call",
]

# Columns zeroed on synthetic rows added by cap_spike (so new rows don't double-count).
CAP_SPIKE_ZERO_COLUMNS = [
    "cost_dialog",
    "cost_task",
    "total_cost_tasks",
    "cost_classification",
    "cost_qc",
    "cost_check_list",
    "total_cost_classifications",
    "cost_amocrm_call",
    "total_cost",
    "has_tasks",
    "has_classifications",
    "has_amocrm_call",
    "num_services",
    "has_both",
    "n_tasks",
    "n_classifications",
    "n_amocrm_calls",
]


# ---------------------------
# Helpers
# ---------------------------


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _date_range(start: date, duration_days: int) -> tuple[date, date]:
    if duration_days <= 0:
        raise ValueError("duration_days must be positive")
    end = start + timedelta(days=duration_days - 1)
    return start, end


def _resolve_days(window_cfg: dict[str, Any]) -> list[date]:
    """Resolve window to an explicit list of dates."""
    if window_cfg.get("dates"):
        raw = window_cfg["dates"]
        if not isinstance(raw, list):
            raise ValueError("window.dates must be a list of ISO date strings")
        days = sorted({_parse_date(str(x)) for x in raw})
        if not days:
            raise ValueError("window.dates must be non-empty")
        return days

    start = _parse_date(str(window_cfg["start_date"]))
    duration_days = int(window_cfg.get("duration_days", 3))
    start_date, end_date = _date_range(start, duration_days)
    return [d.date() for d in pd.date_range(start_date, end_date)]


def _select_accounts(
    account_ids: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> list[int]:
    """Select account IDs per config: 'all' or 'random_k'."""
    mode = cfg.get("mode", "random_k")
    if mode == "all":
        return [int(x) for x in account_ids]
    if mode == "random_k":
        k = int(cfg.get("k", 10))
        k = min(k, len(account_ids))
        if k <= 0:
            return []
        chosen = rng.choice(account_ids, size=k, replace=False)
        return [int(x) for x in chosen]
    raise ValueError(f"Unsupported account_selection mode: {mode}")


def _find_summary_row_idx(df_day: pd.DataFrame) -> int:
    """Pick a row that holds totals (prefer total_cost>0)."""
    if "total_cost" in df_day.columns:
        nonzero = df_day[df_day["total_cost"] > 0]
        if len(nonzero) > 0:
            return int(nonzero.index[0])
        return int(df_day["total_cost"].idxmax())
    return int(df_day.index[0])


def _recompute_derived_fields(row: pd.Series) -> pd.Series:
    """Keep invariants consistent after injection."""

    def g(name: str, default: float = 0.0) -> float:
        v = row.get(name, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def gi(name: str, default: int = 0) -> int:
        v = row.get(name, default)
        try:
            return int(v)
        except Exception:
            return int(default)

    cost_dialog = g("cost_dialog")
    cost_task = g("cost_task")
    cost_classification = g("cost_classification")
    cost_qc = g("cost_qc")
    cost_check_list = g("cost_check_list")
    cost_amocrm_call = g("cost_amocrm_call")

    row["total_cost_tasks"] = float(cost_task)
    row["total_cost_classifications"] = float(cost_classification + cost_qc + cost_check_list)

    row["total_cost"] = float(
        cost_dialog + row["total_cost_tasks"] + row["total_cost_classifications"] + cost_amocrm_call
    )

    n_tasks = gi("n_tasks")
    n_classifications = gi("n_classifications")
    n_calls = gi("n_amocrm_calls")

    has_tasks = int((n_tasks > 0) or (row["total_cost_tasks"] > 0))
    has_classifications = int((n_classifications > 0) or (row["total_cost_classifications"] > 0))
    has_calls = int((n_calls > 0) or (cost_amocrm_call > 0))

    row["has_tasks"] = has_tasks
    row["has_classifications"] = has_classifications
    row["has_amocrm_call"] = has_calls

    num_services = has_tasks + has_classifications + has_calls
    row["num_services"] = int(num_services)
    row["has_both"] = int(has_tasks == 1 and has_classifications == 1)

    return row


def _make_chat_id(rng: np.random.Generator, existing: set[str]) -> str:
    for _ in range(10000):
        cid = "".join(str(int(x)) for x in rng.integers(0, 10, size=11))
        if cid not in existing:
            existing.add(cid)
            return cid
    cid = f"inj{len(existing)}"
    existing.add(cid)
    return cid


def _to_iso_day_map(m: dict[date, list[int]]) -> dict[str, list[int]]:
    return {d.isoformat(): [int(x) for x in xs] for d, xs in m.items()}


def _union_accounts(m: dict[date, list[int]]) -> list[int]:
    out: set[int] = set()
    for xs in m.values():
        out.update(int(x) for x in xs)
    return sorted(out)


def _eligible_accounts_for_cost_component(
    df: pd.DataFrame,
    day: date,
    cols: list[str],
) -> np.ndarray:
    """Accounts that have >0 sum(cols) on that day."""
    df_day = df[df["date"] == day]
    if df_day.empty:
        return np.array([], dtype="int64")
    cols2 = [c for c in cols if c in df_day.columns]
    if not cols2:
        return np.array([], dtype="int64")
    agg = df_day.groupby("account_id", as_index=True)[cols2].sum()
    elig = agg[agg.sum(axis=1) > 0].index.to_numpy()
    return elig.astype("int64")


def _eligible_accounts_any_row(df: pd.DataFrame, day: date) -> np.ndarray:
    df_day = df[df["date"] == day]
    if df_day.empty:
        return np.array([], dtype="int64")
    return df_day["account_id"].dropna().astype("Int64").unique().to_numpy().astype("int64")


# ---------------------------
# Injection types
# ---------------------------


def _apply_cost_spike(
    df: pd.DataFrame,
    requested_by_day: dict[date, list[int]],
    days: Iterable[date],
    params: dict[str, Any],
) -> dict[str, Any]:
    component = str(params["component"])
    magnitude = float(params["magnitude"])
    mult = 1.0 + magnitude

    # If set, we will *create* a non-zero base before multiplying.
    # This is useful if you want very prominent anomalies and don't want
    # "selected but ineffective" account-days.
    min_base_cost = float(params.get("min_base_cost", 0.0))

    # Where to apply the spike:
    # - 'summary_row': only one (preferred) row per account-day (backward compatible)
    # - 'all_rows': multiply costs for every chat row in the account-day (more prominent)
    apply_to = str(params.get("apply_to", "summary_row")).lower()

    field_map: dict[str, list[str]] = {
        "dialog": ["cost_dialog"],
        "tasks": ["cost_task"],
        "classifications": ["cost_classification", "cost_qc", "cost_check_list"],
        "amocrm_call": ["cost_amocrm_call"],
        "all": COST_COLUMNS,
    }
    if component not in field_map:
        raise ValueError(f"Unsupported cost_spike component: {component}")

    cols = [c for c in field_map[component] if c in df.columns]

    effective_by_day: dict[date, list[int]] = defaultdict(list)
    ineffective_by_day: dict[date, list[int]] = defaultdict(list)

    changed_account_days = 0

    for day in tqdm(list(days), desc="Applying cost spike", unit="day"):
        accs = requested_by_day.get(day, [])
        if not accs:
            continue

        for acc in accs:
            mask = (df["account_id"].astype("Int64") == acc) & (df["date"] == day)
            df_day = df.loc[mask]
            if df_day.empty:
                ineffective_by_day[day].append(int(acc))
                continue

            if apply_to not in {"summary_row", "all_rows"}:
                raise ValueError("cost_spike.params.apply_to must be 'summary_row' or 'all_rows'")

            if apply_to == "summary_row":
                idxs = [int(_find_summary_row_idx(df_day))]
            else:
                idxs = [int(i) for i in df_day.index.tolist()]

            # compute before (sum across selected rows)
            if cols:
                before = float(df.loc[idxs, cols].astype(float).to_numpy().sum())
            else:
                before = 0.0

            # ensure base if requested and everything is zero
            if cols and before <= 0.0 and min_base_cost > 0.0:
                per = min_base_cost / float(len(cols))
                summary_idx = int(_find_summary_row_idx(df_day))
                for c in cols:
                    cur = float(df.at[summary_idx, c])
                    if cur <= 0.0:
                        df.at[summary_idx, c] = per

            # apply multiplication on selected rows
            if cols:
                for c in cols:
                    df.loc[idxs, c] = df.loc[idxs, c].astype(float) * mult

            # recompute totals/flags on modified rows
            for idx in idxs:
                df.loc[idx] = _recompute_derived_fields(df.loc[idx])

            if cols:
                after = float(df.loc[idxs, cols].astype(float).to_numpy().sum())
            else:
                after = 0.0

            if after > before + 1e-12:
                effective_by_day[day].append(int(acc))
                changed_account_days += 1
            else:
                ineffective_by_day[day].append(int(acc))

    return {
        "changed_account_days": int(changed_account_days),
        "effective_accounts_by_day": _to_iso_day_map(effective_by_day),
        "ineffective_accounts_by_day": _to_iso_day_map(ineffective_by_day),
    }


def _apply_volume_spike(
    df: pd.DataFrame,
    requested_by_day: dict[date, list[int]],
    days: Iterable[date],
    params: dict[str, Any],
) -> dict[str, Any]:
    component = str(params["component"])
    magnitude = float(params["magnitude"])
    factor = 1.0 + magnitude
    min_base = int(params.get("min_base_count", 1))

    count_cost_map: dict[str, Any] = {
        "tasks": ("n_tasks", ["cost_task"]),
        "classifications": (
            "n_classifications",
            ["cost_classification", "cost_qc", "cost_check_list"],
        ),
        "amocrm_call": ("n_amocrm_calls", ["cost_amocrm_call"]),
        "all": [
            ("n_tasks", ["cost_task"]),
            (
                "n_classifications",
                ["cost_classification", "cost_qc", "cost_check_list"],
            ),
            ("n_amocrm_calls", ["cost_amocrm_call"]),
        ],
    }
    if component not in count_cost_map:
        raise ValueError(f"Unsupported volume_spike component: {component}")

    entries = count_cost_map[component]
    if not isinstance(entries, list):
        entries = [entries]

    effective_by_day: dict[date, list[int]] = defaultdict(list)
    ineffective_by_day: dict[date, list[int]] = defaultdict(list)

    changed_account_days = 0

    for day in tqdm(list(days), desc="Applying volume spike", unit="day"):
        accs = requested_by_day.get(day, [])
        if not accs:
            continue

        for acc in accs:
            mask = (df["account_id"].astype("Int64") == acc) & (df["date"] == day)
            df_day = df.loc[mask]
            if df_day.empty:
                ineffective_by_day[day].append(int(acc))
                continue

            idx = _find_summary_row_idx(df_day)

            changed = False
            for count_col, cost_cols in entries:
                old_n = int(df.at[idx, count_col]) if count_col in df.columns else 0

                if old_n <= 0:
                    old_n = min_base
                    if count_col in df.columns:
                        if int(df.at[idx, count_col]) != old_n:
                            changed = True
                        df.at[idx, count_col] = old_n

                new_n = int(np.ceil(old_n * factor))
                if count_col in df.columns:
                    if int(df.at[idx, count_col]) != new_n:
                        changed = True
                    df.at[idx, count_col] = new_n

                # scale relevant costs roughly proportional to count change
                scale = new_n / max(old_n, 1)
                for col in cost_cols:
                    if col in df.columns:
                        before = float(df.at[idx, col])
                        after = before * scale
                        if abs(after - before) > 1e-12:
                            changed = True
                        df.at[idx, col] = after

            df.loc[idx] = _recompute_derived_fields(df.loc[idx])

            if changed:
                effective_by_day[day].append(int(acc))
                changed_account_days += 1
            else:
                ineffective_by_day[day].append(int(acc))

    return {
        "changed_account_days": int(changed_account_days),
        "effective_accounts_by_day": _to_iso_day_map(effective_by_day),
        "ineffective_accounts_by_day": _to_iso_day_map(ineffective_by_day),
    }


def _apply_cap_spike(
    df: pd.DataFrame,
    requested_by_day: dict[date, list[int]],
    days: Iterable[date],
    params: dict[str, Any],
    max_concurrent: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    exceed_by = int(params.get("exceed_by", 10))
    if exceed_by <= 0:
        raise ValueError("cap_spike.params.exceed_by must be positive")

    existing_chat_ids = set(df["chat_id"].astype(str).tolist())
    new_rows: list[dict[str, Any]] = []

    added_rows = 0
    added_rows_by_day: dict[date, int] = defaultdict(int)
    effective_by_day: dict[date, list[int]] = defaultdict(list)

    for day in tqdm(list(days), desc="Applying cap spike", unit="day"):
        accs = requested_by_day.get(day, [])
        if not accs:
            continue

        for acc in accs:
            mask = (df["account_id"].astype("Int64") == acc) & (df["date"] == day)
            df_day = df.loc[mask]
            if df_day.empty:
                continue

            current = len(df_day)
            target = max_concurrent + exceed_by
            if current >= target:
                continue

            to_add = target - current
            summary_idx = _find_summary_row_idx(df_day)

            dialog_per = float(params.get("dialog_cost_per_extra_chat", 0.0))
            mult = params.get("dialog_cost_multiplier")

            if dialog_per > 0:
                extra_total = dialog_per * to_add
                per_col = extra_total / max(len(COST_COLUMNS), 1)
                for col in COST_COLUMNS:
                    if col in df.columns:
                        df.at[summary_idx, col] = float(df.at[summary_idx, col]) + per_col
            elif mult is not None:
                m = float(mult)
                factor = 1.0 + m * to_add
                for col in COST_COLUMNS:
                    if col in df.columns:
                        df.at[summary_idx, col] = float(df.at[summary_idx, col]) * factor

            df.loc[summary_idx] = _recompute_derived_fields(df.loc[summary_idx])

            template = df_day.iloc[0].copy()
            template["account_id"] = acc
            template["date"] = day

            for _ in range(to_add):
                row = template.copy()
                row["chat_id"] = _make_chat_id(rng, existing_chat_ids)
                for col in CAP_SPIKE_ZERO_COLUMNS:
                    if col in df.columns:
                        row[col] = 0
                new_rows.append(row.to_dict())
                added_rows += 1
                added_rows_by_day[day] += 1

            effective_by_day[day].append(int(acc))

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return (
        {
            "added_rows": int(added_rows),
            "added_rows_by_day": {d.isoformat(): int(v) for d, v in added_rows_by_day.items()},
            "effective_accounts_by_day": _to_iso_day_map(effective_by_day),
            "cap": int(max_concurrent),
            "exceed_by": int(exceed_by),
        },
        df,
    )


# ---------------------------
# Main
# ---------------------------


def inject_anomalies(
    input_csv: str,
    config_path: str,
    output_csv: str,
    labels_jsonl: str,
) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    base_seed = int(cfg.get("base_seed", 123))

    df = load_csv(input_csv)

    required_cols = {"account_id", "chat_id", "date", "total_cost"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns for injection: {sorted(missing)}")

    max_concurrent = int(cfg["dataset"]["max_concurrent_per_account"])

    account_ids_all = df["account_id"].dropna().astype("Int64").unique().to_numpy()

    labels_out: list[dict[str, Any]] = []
    df_out = df.copy()

    scenarios = cfg.get("scenarios", [])
    for i, sc in enumerate(tqdm(scenarios, desc="Injecting scenarios", unit="scenario"), start=1):
        scenario_id = str(sc["scenario_id"])
        anomaly_type = str(sc["anomaly_type"])
        seed = int(sc.get("seed", base_seed + i))
        rng = np.random.default_rng(seed)

        # window (explicit days)
        w_cfg = sc.get("window", cfg.get("defaults", {}).get("window", {"duration_days": 3}))
        days = _resolve_days(w_cfg)
        start_date, end_date = min(days), max(days)

        params = sc.get("params", {})
        driver_template = sc.get("driver_template", "")

        # account selection (optionally per-day)
        acct_cfg = sc.get(
            "account_selection",
            cfg.get("defaults", {}).get("account_selection", {"mode": "random_k", "k": 10}),
        )
        per_day = bool(acct_cfg.get("per_day", False))

        requested_by_day: dict[date, list[int]] = {}

        if per_day:
            # Choose eligible accounts each day
            if anomaly_type == "cost_spike":
                comp = str(params.get("component", "all"))
                field_map = {
                    "dialog": ["cost_dialog"],
                    "tasks": ["cost_task"],
                    "classifications": [
                        "cost_classification",
                        "cost_qc",
                        "cost_check_list",
                    ],
                    "amocrm_call": ["cost_amocrm_call"],
                    "all": COST_COLUMNS,
                }
                cols = field_map.get(comp, COST_COLUMNS)
                for d in days:
                    elig = _eligible_accounts_for_cost_component(df_out, d, cols)
                    requested_by_day[d] = _select_accounts(elig, acct_cfg, rng)

            elif anomaly_type == "volume_spike":
                # eligibility: any row on that day (volume spike can force min_base_count)
                for d in days:
                    elig = _eligible_accounts_any_row(df_out, d)
                    requested_by_day[d] = _select_accounts(elig, acct_cfg, rng)

            elif anomaly_type == "cap_spike":
                for d in days:
                    elig = _eligible_accounts_any_row(df_out, d)
                    requested_by_day[d] = _select_accounts(elig, acct_cfg, rng)

            else:
                raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")
        else:
            # One global account sample, applied to all days
            accounts = _select_accounts(account_ids_all, acct_cfg, rng)
            requested_by_day = dict.fromkeys(days, accounts)

        requested_union = _union_accounts(requested_by_day)

        event_id = f"{scenario_id}__{seed}"

        # Apply
        meta: dict[str, Any]
        affected_services: list[str]

        if anomaly_type == "cost_spike":
            affected_services = [str(params.get("component", "unknown"))]
            meta = _apply_cost_spike(df_out, requested_by_day, days, params)

        elif anomaly_type == "volume_spike":
            affected_services = [str(params.get("component", "unknown"))]
            meta = _apply_volume_spike(df_out, requested_by_day, days, params)

        elif anomaly_type == "cap_spike":
            affected_services = ["concurrency"]
            meta, df_out = _apply_cap_spike(
                df_out, requested_by_day, days, params, max_concurrent, rng
            )

        else:
            raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")

        effective_by_day_iso = meta.get("effective_accounts_by_day", {})
        effective_union: list[int] = sorted(
            {int(a) for xs in effective_by_day_iso.values() for a in xs}
        )

        label = {
            "event_id": event_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "anomaly_type": anomaly_type,
            "affected_services": affected_services,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": [d.isoformat() for d in days],
            "driver_template": driver_template,
            "account_selection": acct_cfg,
            # requested vs effective
            "requested_accounts": requested_union,
            "requested_accounts_by_day": _to_iso_day_map(requested_by_day),
            "n_requested_accounts_union": int(len(requested_union)),
            "effective_accounts": effective_union,
            "n_effective_accounts_union": int(len(effective_union)),
            # legacy fields kept for backward compatibility
            "n_accounts": int(len(requested_union)),
            "accounts_sample": requested_union[:5],
            "affected_accounts": requested_union,
            # params/meta
            "params": params,
            "meta": meta,
        }

        labels_out.append(label)

    # Write outputs
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(labels_jsonl).parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(output_csv, index=False)

    with open(labels_jsonl, "w", encoding="utf-8") as f:
        for obj in labels_out:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inject anomalies into synthetic cost dataset and emit JSONL labels (v2)."
    )
    ap.add_argument("--input", required=True, help="Path to base synthetic CSV")
    ap.add_argument("--config", required=True, help="Path to anomaly scenarios YAML")
    ap.add_argument("--output", required=True, help="Path to output CSV with anomalies")
    ap.add_argument("--labels", required=True, help="Path to output labels JSONL")
    args = ap.parse_args()

    inject_anomalies(
        input_csv=args.input,
        config_path=args.config,
        output_csv=args.output,
        labels_jsonl=args.labels,
    )


if __name__ == "__main__":
    main()
