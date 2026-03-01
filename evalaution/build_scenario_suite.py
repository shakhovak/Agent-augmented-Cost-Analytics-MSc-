#!/usr/bin/env python3

"""
Scenario suite pipeline (orchestrator)

Steps per anomaly event (labels.jsonl):
1) Build POSITIVE facts aligned to the anomaly window (forced dates).
2) Generate NEGATIVE cases (unique IDs per scenario) by calling generate_negative_cases.py.
3) Run LLM to add question variants to positives by calling generate_question_variants.py.
4) Join positives+negatives into a per-scenario "suite.jsonl".
5) Write scenario_config.json + labels.json for traceability.

This script does NOT evaluate answers; it produces the suite inputs.
(Then you run your eval script on the generated suite files.)

Assumptions:
- Dataset has only 2025 dates.
- labels.jsonl events contain: event_id, scenario_id, anomaly_type, start_date, end_date (YYYY-MM-DD).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- IO helpers ----------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def iso_to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def to_iso(d: date) -> str:
    return d.isoformat()


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def run_cmd(cmd: list[str]) -> None:
    env = os.environ.copy()

    # Ensure repo root (and src/) are on PYTHONPATH so `import cost_agent_mvp` works
    add_paths = [str(REPO_ROOT)]
    if SRC_DIR.exists():
        add_paths.append(str(SRC_DIR))

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(add_paths + ([existing] if existing else []))

    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),  # force repo root
        env=env,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{p.stdout}\n"
            f"  stderr:\n{p.stderr}\n"
        )


# ---------- KPI helpers (aligned with your CSV + fact generator logic) ----------

PRICE_PER_ACCOUNT_DAY_USD = 100.0
COST_CAP_USD = 30.0

# "services-only" columns (not dialog)
SERVICE_ONLY_COLS = [
    "total_cost_tasks",
    "total_cost_classifications",
    "cost_amocrm_call",
]
COMPONENTS = [
    ("dialog", "cost_dialog"),
    ("tasks", "total_cost_tasks"),
    ("classifications", "total_cost_classifications"),
    ("amocrm_call", "cost_amocrm_call"),
]


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.date
    df["account_id"] = pd.to_numeric(df["account_id"], errors="raise").astype("int64")

    for col in ["total_cost", "cost_dialog", *SERVICE_ONLY_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def anchor_day_from_df(df: pd.DataFrame) -> date:
    return max(df["date"])


def _service_row_total(sub: pd.DataFrame) -> pd.Series:
    return sub[SERVICE_ONLY_COLS].sum(axis=1)


def _active_service_rows(sub: pd.DataFrame) -> pd.DataFrame:
    if sub.empty:
        return sub.copy()
    s = _service_row_total(sub)
    return sub.loc[s > 0].copy()


def total_cost_services_day(df: pd.DataFrame, day: date) -> float:
    sub = df[df["date"] == day]
    sub = _active_service_rows(sub)
    if sub.empty:
        return 0.0
    return float(_service_row_total(sub).sum())


def per_account_service_costs(df: pd.DataFrame, day: date) -> pd.Series:
    sub = df[df["date"] == day]
    sub = _active_service_rows(sub)
    if sub.empty:
        return pd.Series(dtype=float)
    row_total = _service_row_total(sub)
    per_acc = row_total.groupby(sub["account_id"]).sum()
    per_acc = per_acc[per_acc > 0]
    return per_acc


def active_service_accounts(df: pd.DataFrame, day: date) -> set[int]:
    sub = df[df["date"] == day]
    sub = _active_service_rows(sub)
    if sub.empty:
        return set()
    return set(sub["account_id"].astype("int64").unique().tolist())


def component_totals_for_day(df: pd.DataFrame, day: date) -> dict[str, float]:
    sub = df[df["date"] == day]
    totals: dict[str, float] = {}
    for comp_name, col in COMPONENTS:
        totals[comp_name] = float(sub[col].sum())
    totals["_services_total"] = float(sum(totals.values()))
    return totals


def per_account_day_total_cost(df: pd.DataFrame, day: date) -> pd.Series:
    sub = df[df["date"] == day]
    return sub.groupby("account_id", as_index=True)["total_cost"].sum()


def max_total_cost_services_day_in_window(
    df: pd.DataFrame, start: date, end: date
) -> tuple[date, float]:
    # compute daily service totals inside window; tie-breaker earliest date
    days = sorted({d for d in df["date"].unique().tolist() if start <= d <= end})
    if not days:
        return start, 0.0
    daily = [(d, total_cost_services_day(df, d)) for d in days]
    max_day, max_val = sorted(daily, key=lambda x: (-x[1], x[0]))[0]
    return max_day, float(max_val)


def churn_stats(df: pd.DataFrame, day: date) -> tuple[date, int, int, int]:
    prev = day - timedelta(days=1)
    users_today = active_service_accounts(df, day)
    users_prev = active_service_accounts(df, prev)
    new_users = int(len(users_today - users_prev))
    churned = int(len(users_prev - users_today))
    net = int(len(users_today) - len(users_prev))
    return prev, new_users, churned, net


# ---------- Fact builder (scenario-scoped) ----------


def build_positive_facts_for_event(
    *,
    df: pd.DataFrame,
    csv_path: str,
    anchor_day: date,
    seed: int,
    event: dict[str, Any],
    start_id: int,
) -> list[dict[str, Any]]:
    """
    Build the 10 MVP fact types, but pinned to the anomaly window.
    Uses probe_day=end_date for single-day facts.
    """
    start = iso_to_date(event["start_date"])
    end = iso_to_date(event["end_date"])
    probe_day = end

    mask_window = (df["date"] >= start) & (df["date"] <= end)
    sum_total_window = float(df.loc[mask_window, "total_cost"].sum())

    sum_total_day = float(df.loc[df["date"] == probe_day, "total_cost"].sum())
    sum_services_day = float(total_cost_services_day(df, probe_day))

    # top service (by component totals, including dialog in this helper; matches your generator logic)
    totals = component_totals_for_day(df, probe_day)
    services_total = totals["_services_total"]
    comps = [(k, v) for k, v in totals.items() if k != "_services_total"]
    top_name, top_cost = max(comps, key=lambda kv: kv[1]) if comps else ("none", 0.0)
    top_share = float(top_cost / services_total) if services_total > 0 else 0.0

    # top account by TOTAL cost
    by_acc = per_account_day_total_cost(df, probe_day)
    by_acc = by_acc[by_acc > 0]
    if by_acc.empty:
        top_acc = 0
        top_acc_cost = 0.0
    else:
        top_acc = int(by_acc.idxmax())
        top_acc_cost = float(by_acc.loc[top_acc])

    # p95 of account-day TOTAL cost
    if by_acc.empty:
        p95 = 0.0
    else:
        p95 = float(np.percentile(by_acc.values, 95))

    # margin / cap (services-only per account)
    per_acc_srv = per_account_service_costs(df, probe_day)
    active = int(len(per_acc_srv))
    if active == 0:
        avg_margin = float(PRICE_PER_ACCOUNT_DAY_USD)
        pct_above = 0.0
    else:
        avg_cost = float(per_acc_srv.sum() / active)
        avg_margin = float(PRICE_PER_ACCOUNT_DAY_USD) - avg_cost
        pct_above = float((per_acc_srv > float(COST_CAP_USD)).mean())

    # churn
    prev_day, new_users, churned, net = churn_stats(df, probe_day)

    # max services day in window
    max_day, max_val = max_total_cost_services_day_in_window(df, start, end)

    # Build records (schema aligned with your fact generator)
    next_id = start_id
    base_fingerprint = {
        "csv_path": csv_path,
        "anchor_day": to_iso(anchor_day),
        "seed": seed,
    }

    def mk(rec: dict[str, Any]) -> dict[str, Any]:
        nonlocal next_id
        rec = dict(rec)
        rec["id"] = next_id
        next_id += 1
        rec["anchor_day"] = to_iso(anchor_day)
        rec["csv_fingerprint"] = base_fingerprint
        rec.setdefault("question_variants", [])
        return rec

    facts: list[dict[str, Any]] = []

    facts.append(
        mk(
            {
                "type": "total_cost_window",
                "params": {"start": to_iso(start), "end": to_iso(end)},
                "targets": [
                    {
                        "key": "sum_total_cost",
                        "value": sum_total_window,
                        "tol_abs": 0.01,
                    }
                ],
                "allowed_numbers": [to_iso(start), to_iso(end)],
                "metadata": {"event_id": event["event_id"]},
            }
        )
    )

    facts.append(
        mk(
            {
                "type": "total_cost_single_day",
                "params": {"day": to_iso(probe_day)},
                "targets": [{"key": "sum_total_cost", "value": sum_total_day, "tol_abs": 0.01}],
                "allowed_numbers": [to_iso(probe_day)],
                "metadata": {"event_id": event["event_id"]},
            }
        )
    )

    facts.append(
        mk(
            {
                "type": "total_cost_services_single_day",
                "params": {"day": to_iso(probe_day)},
                "targets": [
                    {
                        "key": "total_cost_services",
                        "value": sum_services_day,
                        "tol_abs": 0.01,
                    }
                ],
                "allowed_numbers": [to_iso(probe_day)],
                "metadata": {"event_id": event["event_id"]},
            }
        )
    )

    # Important: do NOT require service token (to avoid leaking the answer)
    facts.append(
        mk(
            {
                "type": "top_service_by_cost_single_day",
                "params": {"day": to_iso(probe_day)},
                "metadata": {
                    "expected_top_service": top_name,
                    "event_id": event["event_id"],
                },
                "targets": [
                    {
                        "key": "top_service_cost",
                        "value": float(top_cost),
                        "tol_abs": 0.01,
                    },
                    {
                        "key": "top_service_share",
                        "value": float(top_share),
                        "tol_abs": 1e-4,
                    },
                ],
                "allowed_numbers": [to_iso(probe_day)],
            }
        )
    )

    # Important: do NOT require account id token (user shouldn't know it a priori)
    facts.append(
        mk(
            {
                "type": "top_account_by_total_cost_single_day",
                "params": {"day": to_iso(probe_day)},
                "metadata": {
                    "expected_account_id": top_acc,
                    "event_id": event["event_id"],
                },
                "targets": [
                    {
                        "key": "top_account_total_cost",
                        "value": float(top_acc_cost),
                        "tol_abs": 0.01,
                    }
                ],
                "allowed_numbers": [to_iso(probe_day)],
            }
        )
    )

    facts.append(
        mk(
            {
                "type": "p95_account_day_total_cost_single_day",
                "params": {"day": to_iso(probe_day)},
                "targets": [
                    {
                        "key": "p95_account_day_total_cost",
                        "value": float(p95),
                        "tol_abs": 0.01,
                    }
                ],
                "allowed_numbers": [to_iso(probe_day), "95"],
                "metadata": {"event_id": event["event_id"]},
            }
        )
    )

    facts.append(
        mk(
            {
                "type": "avg_margin_per_active_account_day_single_day",
                "params": {"day": to_iso(probe_day)},
                "targets": [
                    {
                        "key": "avg_margin_per_account_day",
                        "value": float(avg_margin),
                        "tol_abs": 0.01,
                    }
                ],
                "allowed_numbers": [to_iso(probe_day)],
                "metadata": {
                    "event_id": event["event_id"],
                    "price_usd": PRICE_PER_ACCOUNT_DAY_USD,
                },
            }
        )
    )

    facts.append(
        mk(
            {
                "type": "pct_accounts_above_cost_cap_single_day",
                "params": {"day": to_iso(probe_day)},
                "metadata": {
                    "cap_usd": float(COST_CAP_USD),
                    "event_id": event["event_id"],
                },
                "targets": [
                    {
                        "key": "pct_accounts_above_cap",
                        "value": float(pct_above),
                        "tol_abs": 1e-4,
                    }
                ],
                "allowed_numbers": [to_iso(probe_day)],
            }
        )
    )

    facts.append(
        mk(
            {
                "type": "user_churn_single_day_vs_prev_day",
                "params": {"day": to_iso(probe_day), "prev_day": to_iso(prev_day)},
                "targets": [
                    {"key": "new_users", "value": new_users, "tol_abs": 0.0},
                    {"key": "churned_users", "value": churned, "tol_abs": 0.0},
                    {"key": "net_user_change", "value": net, "tol_abs": 0.0},
                ],
                "allowed_numbers": [to_iso(probe_day), to_iso(prev_day)],
                "metadata": {"event_id": event["event_id"]},
            }
        )
    )

    facts.append(
        mk(
            {
                "type": "max_total_cost_services_day_in_window",
                "params": {"start": to_iso(start), "end": to_iso(end)},
                "metadata": {"max_day": to_iso(max_day), "event_id": event["event_id"]},
                "targets": [
                    {
                        "key": "max_total_cost_services",
                        "value": float(max_val),
                        "tol_abs": 0.01,
                    }
                ],
                "allowed_numbers": [to_iso(start), to_iso(end), to_iso(max_day)],
            }
        )
    )

    return facts


# ---------- Orchestrator main ----------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a full scenario suite pipeline (facts + negatives + LLM questions)."
    )
    ap.add_argument("--csv", required=True, help="Injected CSV (2025) with anomalies")
    ap.add_argument("--labels", required=True, help="labels.jsonl (anomaly events)")
    ap.add_argument("--out-root", default="outputs/scenario_suites", help="Output root folder")
    ap.add_argument("--suite-id", default="mvp_v0", help="Suite id (folder name)")
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed stored in csv_fingerprint for reproducibility",
    )

    # Existing scripts (reuse)
    ap.add_argument(
        "--negative-script",
        default="evalaution/generate_negative_cases.py",
        help="Path to generate_negative_cases.py",
    )
    ap.add_argument(
        "--question-script",
        default="evalaution/generate_question_variants.py",
        help="Path to generate_question_variants.py",
    )

    # Question generation params
    ap.add_argument(
        "--prompt",
        required=True,
        help="Prompt file for question generation (prompt_for_questions.txt)",
    )
    ap.add_argument("--n-variants", type=int, default=3, help="Question variants per positive fact")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=700)
    ap.add_argument(
        "--skip-llm",
        action="store_true",
        help="Build suite but skip LLM question generation step",
    )

    # Limits
    ap.add_argument("--limit", type=int, default=None, help="Limit number of scenarios (debug)")

    args = ap.parse_args()

    out_root = Path(args.out_root) / args.suite_id
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.csv)
    anchor_day = anchor_day_from_df(df)

    events = read_jsonl(Path(args.labels))
    if args.limit is not None:
        events = events[: args.limit]

    manifest: list[dict[str, Any]] = []

    # deterministic ID ranges per scenario
    base_pos_id = 1
    pos_stride = 1000  # up to 1000 facts per scenario (more than enough)
    base_neg_id = 9001
    neg_stride = 100  # negatives script currently yields ~4 cases

    for idx, ev in enumerate(tqdm(events, desc="Building scenarios", unit="scenario"), start=1):
        event_id = str(ev["event_id"])
        scen_dir = out_root / event_id
        qa_dir = scen_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)

        # Write scenario_config + labels snapshot
        scenario_config = {
            "suite_id": args.suite_id,
            "event_id": event_id,
            "scenario_id": ev.get("scenario_id"),
            "anomaly_type": ev.get("anomaly_type"),
            "start_date": ev.get("start_date"),
            "end_date": ev.get("end_date"),
            "driver_template": ev.get("driver_template"),
            "affected_services": ev.get("affected_services"),
            "affected_accounts": ev.get("affected_accounts"),
            "csv_path": str(args.csv),
            "anchor_day": to_iso(anchor_day),
            "seed": args.seed,
        }
        write_json(scen_dir / "scenario_config.json", scenario_config)
        write_json(scen_dir / "labels.json", ev)

        # 1) Positives (facts pinned to anomaly window)
        pos_start_id = base_pos_id + (idx - 1) * pos_stride
        positives = build_positive_facts_for_event(
            df=df,
            csv_path=str(args.csv),
            anchor_day=anchor_day,
            seed=args.seed,
            event=ev,
            start_id=pos_start_id,
        )
        positives_path = qa_dir / "positives.jsonl"
        write_jsonl(positives_path, positives)

        # 2) Negatives (unique IDs per scenario) - call your existing script
        neg_start_id = base_neg_id + (idx - 1) * neg_stride
        negatives_path = qa_dir / "negatives.jsonl"
        run_cmd(
            [
                sys.executable,
                args.negative_script,
                "--out",
                str(negatives_path),
                "--csv-path",
                str(args.csv),
                "--anchor-day",
                to_iso(anchor_day),
                "--seed",
                str(args.seed),
                "--start-id",
                str(neg_start_id),
            ]
        )

        # 3) LLM: add question_variants to positives
        positives_with_q_path = qa_dir / "positives_with_questions.jsonl"
        if args.skip_llm:
            # Just copy positives through
            positives_with_q_path.write_text(
                positives_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        else:
            run_cmd(
                [
                    sys.executable,  # <-- not "python"
                    args.question_script,
                    "--facts",
                    str(positives_path),
                    "--out",
                    str(positives_with_q_path),
                    "--prompt",
                    str(args.prompt),
                    "--n",
                    str(args.n_variants),
                    "--temperature",
                    str(args.temperature),
                    "--max-output-tokens",
                    str(args.max_output_tokens),
                ]
            )

        # 4) Join into a single suite file for convenience (your eval can still read separate files if you prefer)
        suite_path = qa_dir / "suite.jsonl"
        suite_rows: list[dict[str, Any]] = []
        suite_rows += read_jsonl(positives_with_q_path)
        suite_rows += read_jsonl(negatives_path)
        write_jsonl(suite_path, suite_rows)

        # Optional: write a short deterministic scenario description (no LLM)
        desc = (
            f"Scenario {ev.get('scenario_id')} / {event_id}\n"
            f"Type: {ev.get('anomaly_type')}\n"
            f"Window: {ev.get('start_date')} .. {ev.get('end_date')}\n"
            f"Affected services: {ev.get('affected_services')}\n"
            f"Affected accounts: {len(ev.get('affected_accounts') or [])}\n"
            f"Driver template (label): {ev.get('driver_template')}\n"
        )
        (scen_dir / "scenario_description.txt").write_text(desc, encoding="utf-8")

        manifest.append(
            {
                "event_id": event_id,
                "scenario_dir": str(scen_dir),
                "positives": str(positives_with_q_path),
                "negatives": str(negatives_path),
                "suite": str(suite_path),
            }
        )

    write_jsonl(out_root / "manifest.jsonl", manifest)
    print(f"Wrote {len(manifest)} scenarios to {out_root}")
    print(f"Manifest: {out_root / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
