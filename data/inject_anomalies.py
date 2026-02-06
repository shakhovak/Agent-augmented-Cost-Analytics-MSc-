"""Inject configurable anomalies into synthetic cost data and emit JSONL labels.

Supports cost_spike (multiply cost components), volume_spike (scale counts and costs),
and cap_spike (add rows above max_concurrent and optionally scale costs).
Scenario config is read from a YAML file; see data/anomaly_scenarios.yaml.
"""

from __future__ import annotations

import argparse
import json
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
    """Parse ISO date string."""
    return date.fromisoformat(s)


def _date_range(start: date, duration_days: int) -> tuple[date, date]:
    """Return (start, end) inclusive for the given duration."""
    if duration_days <= 0:
        raise ValueError("duration_days must be positive.")
    end = start + timedelta(days=duration_days - 1)
    return start, end


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
        chosen = rng.choice(account_ids, size=k, replace=False)
        return [int(x) for x in chosen]
    raise ValueError(f"Unsupported account_selection mode: {mode}")


def _find_summary_row_idx(df_day: pd.DataFrame) -> int:
    """
    Costs/totals are typically written on one row per account-day.
    Prefer a row that already carries total_cost>0, else max(total_cost), else first row.
    """
    if "total_cost" in df_day.columns:
        nonzero = df_day[df_day["total_cost"] > 0]
        if len(nonzero) > 0:
            return int(nonzero.index[0])
        return int(df_day["total_cost"].idxmax())
    return int(df_day.index[0])


def _recompute_derived_fields(row: pd.Series) -> pd.Series:
    """
    Keep invariants consistent after injection:
    - total_cost_tasks / total_cost_classifications aligned with component costs
    - total_cost equals sum of main components
    - has_* flags consistent with counts/costs
    - num_services and has_both consistent
    """

    # Safe getters
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

    # Components
    cost_dialog = g("cost_dialog")
    cost_task = g("cost_task")
    cost_classification = g("cost_classification")
    cost_qc = g("cost_qc")
    cost_check_list = g("cost_check_list")
    cost_amocrm_call = g("cost_amocrm_call")

    # Totals per service-group
    row["total_cost_tasks"] = float(cost_task)
    row["total_cost_classifications"] = float(cost_classification + cost_qc + cost_check_list)

    # Main total
    row["total_cost"] = float(
        cost_dialog + row["total_cost_tasks"] + row["total_cost_classifications"] + cost_amocrm_call
    )

    # Flags and counts
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
    # 11 digits similar to your sample; ensure uniqueness
    for _ in range(10000):
        cid = "".join(str(int(x)) for x in rng.integers(0, 10, size=11))
        if cid not in existing:
            existing.add(cid)
            return cid
    # fallback
    cid = f"inj{len(existing)}"
    existing.add(cid)
    return cid


# ---------------------------
# Injection types
# ---------------------------


def _apply_cost_spike(
    df: pd.DataFrame,
    accounts: list[int],
    start: date,
    end: date,
    params: dict[str, Any],
) -> dict[str, int]:
    component = params["component"]
    magnitude = float(params["magnitude"])
    mult = 1.0 + magnitude

    field_map = {
        "dialog": ["cost_dialog"],
        "tasks": ["cost_task"],
        "classifications": ["cost_classification", "cost_qc", "cost_check_list"],
        "amocrm_call": ["cost_amocrm_call"],
        "all": COST_COLUMNS,
    }
    if component not in field_map:
        raise ValueError(f"Unsupported cost_spike component: {component}")

    changed_days = 0
    for acc in tqdm(accounts, desc="Applying cost spike", unit="account"):
        mask_acc = df["account_id"].astype("Int64") == acc
        df_acc = df.loc[mask_acc]
        if df_acc.empty:
            continue

        for day in pd.date_range(start, end).date:
            mask_day = mask_acc & (df["date"] == day)
            df_day = df.loc[mask_day]
            if df_day.empty:
                continue

            idx = _find_summary_row_idx(df_day)
            for col in field_map[component]:
                if col in df.columns:
                    df.at[idx, col] = float(df.at[idx, col]) * mult
            # recompute totals/flags on summary row
            df.loc[idx] = _recompute_derived_fields(df.loc[idx])
            changed_days += 1

    return {"changed_account_days": changed_days}


def _apply_volume_spike(
    df: pd.DataFrame,
    accounts: list[int],
    start: date,
    end: date,
    params: dict[str, Any],
) -> dict[str, int]:
    component = params["component"]
    magnitude = float(params["magnitude"])
    factor = 1.0 + magnitude
    min_base = int(params.get("min_base_count", 1))

    count_cost_map = {
        "tasks": ("n_tasks", ["cost_task"]),
        "classifications": (
            "n_classifications",
            ["cost_classification", "cost_qc", "cost_check_list"],
        ),
        "amocrm_call": ("n_amocrm_calls", ["cost_amocrm_call"]),
        "all": [
            ("n_tasks", ["cost_task"]),
            ("n_classifications", ["cost_classification", "cost_qc", "cost_check_list"]),
            ("n_amocrm_calls", ["cost_amocrm_call"]),
        ],
    }
    if component not in count_cost_map:
        raise ValueError(f"Unsupported volume_spike component: {component}")

    entries = count_cost_map[component]
    if not isinstance(entries, list):
        entries = [entries]

    changed_days = 0
    for acc in tqdm(accounts, desc="Applying volume spike", unit="account"):
        mask_acc = df["account_id"].astype("Int64") == acc
        for day in pd.date_range(start, end).date:
            mask_day = mask_acc & (df["date"] == day)
            df_day = df.loc[mask_day]
            if df_day.empty:
                continue

            idx = _find_summary_row_idx(df_day)

            for count_col, cost_cols in entries:
                old_n = int(df.at[idx, count_col]) if count_col in df.columns else 0
                if old_n <= 0:
                    old_n = min_base
                    if count_col in df.columns:
                        df.at[idx, count_col] = old_n

                new_n = int(np.ceil(old_n * factor))
                if count_col in df.columns:
                    df.at[idx, count_col] = new_n

                # scale relevant costs roughly proportional to count change
                scale = new_n / max(old_n, 1)
                for col in cost_cols:
                    if col in df.columns:
                        df.at[idx, col] = float(df.at[idx, col]) * scale

            df.loc[idx] = _recompute_derived_fields(df.loc[idx])
            changed_days += 1

    return {"changed_account_days": changed_days}


def _apply_cap_spike(
    df: pd.DataFrame,
    accounts: list[int],
    start: date,
    end: date,
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

    for acc in tqdm(accounts, desc="Applying cap spike", unit="account"):
        mask_acc = df["account_id"].astype("Int64") == acc
        for day in pd.date_range(start, end).date:
            mask_day = mask_acc & (df["date"] == day)
            df_day = df.loc[mask_day]
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

            # recompute totals/flags after changing costs
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

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return {"added_rows": added_rows, "cap": max_concurrent, "exceed_by": exceed_by}, df


# ---------------------------
# Main
# ---------------------------


def inject_anomalies(
    input_csv: str,
    config_path: str,
    output_csv: str,
    labels_jsonl: str,
) -> None:
    """Load base CSV and YAML config, apply all scenarios, write anomaly CSV and JSONL labels."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    base_seed = int(cfg.get("base_seed", 123))
    # rng_global = np.random.default_rng(base_seed)

    # Load base data
    df = load_csv(input_csv)

    # Ensure expected columns exist
    required_cols = {"account_id", "chat_id", "date", "total_cost"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns for injection: {sorted(missing)}")

    # Scenario config
    max_concurrent = int(cfg["dataset"]["max_concurrent_per_account"])

    account_ids = df["account_id"].dropna().astype("Int64").unique().to_numpy()

    labels_out: list[dict[str, Any]] = []
    df_out = df.copy()

    scenarios = cfg.get("scenarios", [])
    for i, sc in enumerate(tqdm(scenarios, desc="Injecting scenarios", unit="scenario"), start=1):
        scenario_id = str(sc["scenario_id"])
        anomaly_type = str(sc["anomaly_type"])
        seed = int(sc.get("seed", base_seed + i))
        rng = np.random.default_rng(seed)

        # accounts
        acct_cfg = sc.get(
            "account_selection",
            cfg.get("defaults", {}).get("account_selection", {"mode": "random_k", "k": 10}),
        )
        accounts = _select_accounts(account_ids, acct_cfg, rng)

        # window
        w_cfg = sc.get("window", cfg.get("defaults", {}).get("window", {"duration_days": 3}))
        start = _parse_date(str(w_cfg["start_date"]))
        duration_days = int(w_cfg.get("duration_days", 3))
        start_date, end_date = _date_range(start, duration_days)

        params = sc.get("params", {})
        driver_template = sc.get("driver_template", "")

        event_id = f"{scenario_id}__{seed}"

        # Apply
        meta: dict[str, Any] = {}
        affected_services: list[str] = []

        if anomaly_type == "cost_spike":
            affected_services = [params.get("component", "unknown")]
            meta = _apply_cost_spike(df_out, accounts, start_date, end_date, params)
        elif anomaly_type == "volume_spike":
            affected_services = [params.get("component", "unknown")]
            meta = _apply_volume_spike(df_out, accounts, start_date, end_date, params)
        elif anomaly_type == "cap_spike":
            affected_services = ["concurrency"]
            meta, df_out = _apply_cap_spike(
                df_out, accounts, start_date, end_date, params, max_concurrent, rng
            )
        else:
            raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")

        label = {
            "event_id": event_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "anomaly_type": anomaly_type,
            "affected_services": affected_services,  # JSON list (best for JSONL)
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "driver_template": driver_template,
            "account_selection": acct_cfg,
            "n_accounts": len(accounts),
            "accounts_sample": accounts[:5],  # keep labels small but informative
            "affected_accounts": accounts,
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
    """CLI entrypoint: parse args and run inject_anomalies."""
    ap = argparse.ArgumentParser(
        description="Inject anomalies into synthetic cost dataset and emit JSONL labels."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to base synthetic CSV (e.g., data/sample.csv)",
    )
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
