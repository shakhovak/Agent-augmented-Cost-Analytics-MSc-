#!/usr/bin/env python3

"""
eval_scenario_bundle.py

Bundle-level evaluator for scenario suites.

It treats each scenario folder (one labelled anomaly event) as the unit of analysis and produces:
- RQ1-ish: anomaly detection event metrics (hit/delay/false-positive days in a local neighborhood)
- RQ2-ish: QA numeric fidelity + negative (UNSUPPORTED) handling, using your existing QA evaluator
- RQ3-ish: integration/consistency checks that link detector <-> QA outputs inside the same scenario

Expected suite layout (from build_scenario_suite.py):
outputs/<suite_root>/<suite_id>/
  manifest.jsonl
  <event_id>/
    scenario_config.json
    labels.json
    qa/
      suite.jsonl            # positives_with_questions + negatives
    ...

This script will (by default) run the QA evaluator per scenario, writing:
  <event_id>/qa_eval/qa/summary.csv

and will also write:
  <event_id>/scenario_metrics.json

Finally it writes suite-level aggregates:
  <suite_root>/<suite_id>/bundle_results.csv
  <suite_root>/<suite_id>/bundle_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# --- Detector (deterministic) ---
# Assumes you have this module in your repo (as discussed)
from cost_agent_mvp.agent.detector import detect_anomaly_for_end_day

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger("bundle_eval")


# -----------------------------
# Utilities
# -----------------------------
def _true_accounts_for_day(
    label_event: dict[str, Any], day_iso: str, *, label_by_day_field: str
) -> set[int]:
    true_accounts: set[int] = set()

    by_day = label_event.get(label_by_day_field, {})
    if isinstance(by_day, dict):
        xs = by_day.get(day_iso, [])
        if isinstance(xs, list):
            true_accounts = {int(x) for x in xs}

    # fallback: union over all days
    if not true_accounts and isinstance(by_day, dict):
        for xs in by_day.values():
            if isinstance(xs, list):
                true_accounts.update(int(x) for x in xs)

    # last fallback: union-level lists if present
    if not true_accounts:
        for k in ("effective_accounts", "affected_accounts", "requested_accounts"):
            xs = label_event.get(k)
            if isinstance(xs, list):
                true_accounts = {int(x) for x in xs}
                break

    return true_accounts


def _pred_accounts_from_detection(
    det: dict[str, Any], *, anomaly_type: str, top_k: int
) -> tuple[set[int], str]:
    """
    Returns (predicted_accounts_set, source).
    Prefers: flagged_account_ids -> account_decisions -> top_accounts (top_k).
    """
    decision = det.get("decision", {}) if isinstance(det.get("decision", {}), dict) else {}

    # Preferred: full list
    if isinstance(decision.get("flagged_account_ids"), list):
        return (
            {int(x) for x in decision["flagged_account_ids"]},
            "flagged_account_ids",
        )

    # Next: account_decisions flags (debug / include-account-decision)
    if isinstance(decision.get("account_decisions"), list):
        flag_key = f"flag_{anomaly_type}"
        pred: set[int] = set()
        for row in decision["account_decisions"]:
            if not isinstance(row, dict):
                continue
            if row.get(flag_key) is True and "account_id" in row:
                pred.add(int(row["account_id"]))
        return pred, "account_decisions"

    # Fallback: top_accounts (partial) -> evaluate as @K
    top = decision.get("top_accounts", []) or []
    pred = set()
    for row in top[: max(0, int(top_k))]:
        if isinstance(row, dict) and "account_id" in row:
            pred.add(int(row["account_id"]))
    return pred, f"top_accounts@{top_k}"


def _compare_accounts(true_accounts: set[int], pred_accounts: set[int]) -> dict[str, Any]:
    tp = pred_accounts & true_accounts
    fp = pred_accounts - true_accounts
    fn = true_accounts - pred_accounts

    precision = (len(tp) / len(pred_accounts)) if pred_accounts else 0.0
    recall = (len(tp) / len(true_accounts)) if true_accounts else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "n_true": len(true_accounts),
        "n_pred": len(pred_accounts),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_set": tp,
        "fp_set": fp,
        "fn_set": fn,
    }


def _clip_accounts(xs: set[int], max_list: int) -> list[int]:
    return sorted(xs)[: max(0, int(max_list))]


def eval_account_attribution_for_day(
    *,
    repo_root: Path,
    extra_py: list[Path],
    detector_cli_script: str,
    csv_path: str,
    label_event: dict[str, Any],
    anomaly_type: str,
    day_iso: str,
    label_by_day_field: str,
    top_k: int,
    max_list: int,
    detector_cli_args: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    """
    Runs detector CLI with --include-account-decision, then computes account-level precision/recall vs labels.
    Writes detector JSON into out_dir for auditability.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    det_out = out_dir / f"detector_{day_iso}.json"

    cmd = [
        sys.executable,
        str(Path(detector_cli_script)),
        "--csv",
        str(csv_path),
        "--end-day",
        day_iso,
        "--out",
        str(det_out),
        "--include-account-decision",
    ]
    if detector_cli_args:
        cmd += detector_cli_args

    run_cmd(cmd, cwd=repo_root, extra_pythonpath=extra_py)

    det = json.loads(det_out.read_text(encoding="utf-8"))
    true_accounts = _true_accounts_for_day(
        label_event, day_iso, label_by_day_field=label_by_day_field
    )
    pred_accounts, pred_source = _pred_accounts_from_detection(
        det, anomaly_type=anomaly_type, top_k=top_k
    )

    comp = _compare_accounts(true_accounts, pred_accounts)

    return {
        "comparison_day": day_iso,
        "label_by_day_field": label_by_day_field,
        "predicted_source": pred_source,
        "counts": {
            "n_true": comp["n_true"],
            "n_pred": comp["n_pred"],
            "tp": comp["tp"],
            "fp": comp["fp"],
            "fn": comp["fn"],
            "precision": comp["precision"],
            "recall": comp["recall"],
            "f1": comp["f1"],
        },
        "tp_accounts": _clip_accounts(comp["tp_set"], max_list),
        "fp_accounts": _clip_accounts(comp["fp_set"], max_list),
        "fn_accounts": _clip_accounts(comp["fn_set"], max_list),
        "detector_json": str(det_out),
    }


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


def date_to_iso(d: date) -> str:
    return d.isoformat()


def daterange(start: date, end: date) -> list[date]:
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]


def find_repo_root(start: Path) -> Path:
    """
    Search upwards for pyproject.toml to infer repo root.
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    extra_pythonpath: list[Path] | None = None,
) -> None:
    env = os.environ.copy()
    add_paths = []
    if extra_pythonpath:
        add_paths = [str(p) for p in extra_pythonpath if p.exists()]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(add_paths + ([existing] if existing else []))

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  cwd: {cwd}\n"
            f"  stdout:\n{p.stdout}\n"
            f"  stderr:\n{p.stderr}\n"
        )


def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x.dropna()
    if len(x) == 0:
        return 0.0
    return float(x.mean())


def file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


# -----------------------------
# RQ1: detection evaluation
# -----------------------------


@dataclass
class DetectionEventMetrics:
    hit: bool
    first_flag_day: str | None
    delay_days: int | None
    n_flagged_in_window: int
    n_flagged_nearby_outside_window: int
    flagged_days_in_window: list[str]
    flagged_days_nearby: list[str]
    top_driver_in_window: str | None


def eval_detector_for_event(
    *,
    csv_path: str,
    start: date,
    end: date,
    baseline_days: int = 7,
    z_threshold: float = 5.5,
    cost_cap_usd: float = 70.0,
    neighborhood_days: int = 7,
) -> tuple[DetectionEventMetrics, dict[str, Any]]:
    """
    Event-level evaluation: scan each day in label window and record flagged days.
    Also count "nearby false positives" in a local neighborhood around the window:
      [start - neighborhood_days .. end + neighborhood_days] excluding the label window.
    """
    window_days = daterange(start, end)

    cache: dict[str, dict[str, Any]] = {}

    def det_for_day(d: date) -> dict[str, Any]:
        key = d.isoformat()
        if key in cache:
            return cache[key]
        det = detect_anomaly_for_end_day(
            csv_path=csv_path,
            end_day=d,
            baseline_days=baseline_days,
            context_days=7,
            z_threshold=z_threshold,
            cost_cap_usd=cost_cap_usd,
        )
        cache[key] = det
        return det

    flagged_in_window: list[tuple[date, dict[str, Any]]] = []
    for d in window_days:
        det = det_for_day(d)
        dec = det.get("decision") or {}
        if bool(dec.get("anomaly")):
            flagged_in_window.append((d, det))

    hit = len(flagged_in_window) > 0
    if hit:
        first_day = min(d for d, _ in flagged_in_window)
        delay = (first_day - start).days
        first_flag_day = first_day.isoformat()
    else:
        delay = None
        first_flag_day = None

    flagged_days_in_window = [
        d.isoformat() for d, _ in sorted(flagged_in_window, key=lambda x: x[0])
    ]

    # top driver = driver of max severity day inside window
    top_driver = None
    if hit:
        best = max(
            flagged_in_window,
            key=lambda x: float((x[1].get("decision") or {}).get("severity") or 0.0),
        )
        top_driver = str((best[1].get("decision") or {}).get("driver") or "")

    # Nearby scanning
    scan_start = start - timedelta(days=neighborhood_days)
    scan_end = end + timedelta(days=neighborhood_days)
    scan_days = daterange(scan_start, scan_end)

    flagged_nearby: list[str] = []
    n_outside = 0
    for d in scan_days:
        det = det_for_day(d)
        dec = det.get("decision") or {}
        if bool(dec.get("anomaly")):
            flagged_nearby.append(d.isoformat())
            if d < start or d > end:
                n_outside += 1

    metrics = DetectionEventMetrics(
        hit=hit,
        first_flag_day=first_flag_day,
        delay_days=delay,
        n_flagged_in_window=len(flagged_in_window),
        n_flagged_nearby_outside_window=n_outside,
        flagged_days_in_window=flagged_days_in_window,
        flagged_days_nearby=flagged_nearby,
        top_driver_in_window=top_driver,
    )

    debug = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "scan_neighborhood": {
            "start": scan_start.isoformat(),
            "end": scan_end.isoformat(),
        },
        "detector_params": {
            "baseline_days": baseline_days,
            "z_threshold": z_threshold,
            "cost_cap_usd": cost_cap_usd,
            "neighborhood_days": neighborhood_days,
        },
    }
    return metrics, debug


# -----------------------------
# RQ2: QA evaluation (reuse existing evaluator)
# -----------------------------


@dataclass
class QAMetrics:
    positives_em_rate: float
    negatives_refusal_rate: float
    planner_time_window_ok_rate: float
    planner_passed_rate: float
    n_positive_runs: int
    n_negative_runs: int
    qa_summary_csv: str


def load_qa_metrics(summary_csv: Path) -> QAMetrics:
    df = pd.read_csv(summary_csv)

    # Keep only OK runs
    df_ok = df[df["status"] == "OK"].copy()

    # Ensure passed/planner_passed are numeric/bool-safe
    for col in ("passed", "planner_passed", "planner_time_window_ok"):
        if col in df_ok.columns:
            # Handles True/False strings and 0/1
            df_ok[col] = pd.to_numeric(df_ok[col], errors="coerce")

    pos = df_ok[df_ok.get("is_negative", False) == False].copy()  # noqa: E712
    neg = df_ok[df_ok.get("is_negative", False) == True].copy()  # noqa: E712

    positives_em = float(pos["passed"].mean()) if ("passed" in pos.columns and len(pos)) else 0.0
    negatives_refusal = (
        float(neg["passed"].mean()) if ("passed" in neg.columns and len(neg)) else 0.0
    )

    # planner_time_window_ok can be missing or null — compute only if present and has non-null values
    tw_ok = 0.0
    if "planner_time_window_ok" in pos.columns and len(pos):
        tw = pos["planner_time_window_ok"].dropna()
        tw_ok = float(tw.mean()) if len(tw) else 0.0

    planner_passed = 0.0
    if "planner_passed" in pos.columns and len(pos):
        pp = pos["planner_passed"].dropna()
        planner_passed = float(pp.mean()) if len(pp) else 0.0

    return QAMetrics(
        positives_em_rate=positives_em,
        negatives_refusal_rate=negatives_refusal,
        planner_time_window_ok_rate=tw_ok,
        planner_passed_rate=planner_passed,
        n_positive_runs=int(len(pos)),
        n_negative_runs=int(len(neg)),
        qa_summary_csv=str(summary_csv),
    )


# -----------------------------
# RQ3: Integration checks
# -----------------------------


@dataclass
class IntegrationMetrics:
    detector_context_injected_rate: float
    mention_flagged_day_rate: float
    n_checked_answers: int


def detector_injected_in_run(run_dir: Path, marker: str = "Deterministic anomaly detector") -> bool:
    trace = run_dir / "llm_trace.json"
    if not trace.exists():
        return False
    try:
        items = json.loads(trace.read_text(encoding="utf-8"))
    except Exception:
        return False
    # Expect list of {kind, system_preview, user_preview, response_text}
    for it in items:
        if isinstance(it, dict) and it.get("kind") == "analyst":
            user_preview = str(it.get("user_preview") or "")
            if marker in user_preview:
                return True
    return False


def answer_mentions_any_day(answer_path: Path, days: list[str]) -> bool:
    if not answer_path.exists():
        return False
    txt = answer_path.read_text(encoding="utf-8", errors="ignore")
    return any(d in txt for d in days)


def compute_integration_metrics(
    *,
    qa_summary_csv: Path,
    flagged_days_in_window: list[str],
    check_fact_types: Sequence[str] = (
        "total_cost_window",
        "max_total_cost_services_day_in_window",
    ),
) -> IntegrationMetrics:
    """
    Lightweight RQ3 checks:
    1) Was detector context injected into analyst prompt? (via marker in llm_trace.json)
    2) If there are flagged days, does the final answer mention at least one flagged day (for window-like questions)?
    """
    df = pd.read_csv(qa_summary_csv)
    df_ok = df[(df["status"] == "OK") & (not df["is_negative"])].copy()  # noqa: E712

    # Only check window-like fact types (these are the ones where dates should appear)
    df_chk = df_ok[df_ok["fact_type"].isin(check_fact_types)].copy()
    if len(df_chk) == 0:
        return IntegrationMetrics(
            detector_context_injected_rate=0.0,
            mention_flagged_day_rate=0.0,
            n_checked_answers=0,
        )

    injected = 0
    mentioned = 0
    checked = 0

    for _, row in df_chk.iterrows():
        run_dir = Path(str(row["run_dir"]))
        checked += 1

        if detector_injected_in_run(run_dir):
            injected += 1

        # Mention check only meaningful when we actually flagged something
        if flagged_days_in_window:
            ans = run_dir / "final_answer.txt"
            if answer_mentions_any_day(ans, flagged_days_in_window):
                mentioned += 1

    injected_rate = injected / checked if checked else 0.0
    mention_rate = (mentioned / checked) if checked and flagged_days_in_window else 0.0

    return IntegrationMetrics(
        detector_context_injected_rate=float(injected_rate),
        mention_flagged_day_rate=float(mention_rate),
        n_checked_answers=int(checked),
    )


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate scenario bundle (RQ1+RQ2+RQ3) by reading scenario folders."
    )
    # --- RQ1 (account-level) evaluation via detector CLI ---
    ap.add_argument(
        "--account-eval",
        action="store_true",
        help="Compute account precision/recall using detector CLI --include-account-decision",
    )
    ap.add_argument(
        "--detector-cli-script",
        default="src/cost_agent_mvp/cli/run_detector_7d.py",
        help="Path to run_detector_7d.py",
    )
    ap.add_argument(
        "--account-day-mode",
        choices=["start", "end", "first_flag"],
        default="start",
        help="Which day to use for account-level evaluation: label start, label end, or first detected flagged day (fallback to start)",
    )
    ap.add_argument(
        "--label-by-day-field",
        default="requested_accounts_by_day",
        help="Label field holding true accounts per day",
    )
    ap.add_argument(
        "--account-top-k",
        type=int,
        default=100,
        help="If detector output only has top_accounts, evaluate precision@K/recall@K at this K",
    )
    ap.add_argument(
        "--account-max-list",
        type=int,
        default=100,
        help="How many tp/fp/fn account IDs to store in scenario_metrics.json (debug convenience)",
    )
    ap.add_argument(
        "--detector-cli-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to detector CLI (e.g. --z-threshold 4.5 --baseline-days 14). "
        "Usage: --detector-cli-args --arg1 val1 --arg2 val2",
    )
    ap.add_argument(
        "--suite-root",
        required=True,
        help="Path to suite folder (contains manifest.jsonl).",
    )
    ap.add_argument("--semantic-layer-path", required=True, help="Path to semantic layer YAML.")
    ap.add_argument("--prompts-dir", required=True, help="Path to prompts dir (planner/analyst).")
    ap.add_argument(
        "--qa-eval-script",
        default="evalaution/eval_scenarios_fidelity.py",
        help="Path to QA fidelity evaluator script.",
    )
    ap.add_argument(
        "--mode",
        default="llm",
        choices=["llm", "qa"],
        help="Mode passed to QA evaluator.",
    )
    ap.add_argument("--mock-planner", action="store_true")
    ap.add_argument("--mock-analyst", action="store_true")
    ap.add_argument(
        "--skip-qa-eval",
        action="store_true",
        help="Do not rerun QA eval; just aggregate existing outputs.",
    )
    ap.add_argument("--limit", type=int, default=None)

    # Detector params
    ap.add_argument("--baseline-days", type=int, default=7)
    ap.add_argument("--z-threshold", type=float, default=5.5)
    ap.add_argument("--cap", type=float, default=70.0)
    ap.add_argument("--neighborhood-days", type=int, default=7)

    ap.add_argument(
        "--account-days",
        choices=["single", "label_window", "flagged_window"],
        default="single",
        help="Days to evaluate for account attribution.",
    )

    # Output
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Write suite-level CSV here (default: <suite-root>/bundle_results.csv)",
    )
    args = ap.parse_args()

    suite_root = Path(args.suite_root)
    manifest_path = suite_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.jsonl not found: {manifest_path}")

    repo_root = find_repo_root(Path(__file__).resolve())
    extra_py = [repo_root, repo_root / "src"]

    manifest = read_jsonl(manifest_path)
    if args.limit is not None:
        manifest = manifest[: args.limit]

    bundle_rows: list[dict[str, Any]] = []

    for item in tqdm(manifest, desc="Scenarios", unit="scenario"):
        event_id = str(item["event_id"])
        scen_dir = Path(item["scenario_dir"])
        suite_jsonl = Path(item["suite"])

        labels_path = scen_dir / "labels.json"
        cfg_path = scen_dir / "scenario_config.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels.json in {scen_dir}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing scenario_config.json in {scen_dir}")

        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        csv_path = str(cfg.get("csv_path") or "")
        if not csv_path:
            raise ValueError(f"scenario_config.json missing csv_path for {event_id}")

        start = iso_to_date(labels["start_date"])
        end = iso_to_date(labels["end_date"])
        anomaly_type = str(labels.get("anomaly_type") or cfg.get("anomaly_type") or "unknown")
        scenario_id = str(labels.get("scenario_id") or cfg.get("scenario_id") or "unknown")
        logger.info(
            "Scenario %s | type=%s | window=%s..%s",
            event_id,
            anomaly_type,
            labels["start_date"],
            labels["end_date"],
        )
        # ---------- RQ1: detector eval ----------
        det_metrics, det_debug = eval_detector_for_event(
            csv_path=csv_path,
            start=start,
            end=end,
            baseline_days=args.baseline_days,
            z_threshold=args.z_threshold,
            cost_cap_usd=args.cap,
            neighborhood_days=args.neighborhood_days,
        )
        logger.info(
            "RQ1 detector | hit=%s | delay=%s | flagged_in_window=%d | fp_nearby=%d",
            det_metrics.hit,
            det_metrics.delay_days,
            det_metrics.n_flagged_in_window,
            det_metrics.n_flagged_nearby_outside_window,
        )
        write_json(
            scen_dir / "detector_eval.json",
            {"metrics": det_metrics.__dict__, "debug": det_debug},
        )
        # ---------- RQ1b: account-level attribution (optional) ----------
        acct_eval: dict[str, Any] = {}
        if args.account_eval:
            # choose days
            if args.account_days == "single":
                if args.account_day_mode == "start":
                    days = [labels["start_date"]]
                elif args.account_day_mode == "end":
                    days = [labels["end_date"]]
                else:
                    days = [det_metrics.first_flag_day or labels["start_date"]]

            elif args.account_days == "label_window":
                days = [d.isoformat() for d in daterange(start, end)]

            else:  # flagged_window
                days = det_metrics.flagged_days_in_window[:] or [labels["start_date"]]

            per_day = []
            sum_tp = sum_fp = sum_fn = sum_true = sum_pred = 0

            for day_iso in days:
                one = eval_account_attribution_for_day(
                    repo_root=repo_root,
                    extra_py=extra_py,
                    detector_cli_script=args.detector_cli_script,
                    csv_path=csv_path,
                    label_event=labels,
                    anomaly_type=anomaly_type,
                    day_iso=day_iso,
                    label_by_day_field=args.label_by_day_field,
                    top_k=args.account_top_k,
                    max_list=args.account_max_list,
                    detector_cli_args=args.detector_cli_args,
                    out_dir=scen_dir / "detector_accounts",
                )
                per_day.append(one)

                c = one["counts"]
                sum_tp += int(c["tp"])
                sum_fp += int(c["fp"])
                sum_fn += int(c["fn"])
                sum_true += int(c["n_true"])
                sum_pred += int(c["n_pred"])

            prec = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) else 0.0
            rec = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

            acct_eval = {
                "mode": args.account_days,
                "days": days,
                "per_day": per_day,
                "aggregate": {
                    "tp": sum_tp,
                    "fp": sum_fp,
                    "fn": sum_fn,
                    "n_true": sum_true,
                    "n_pred": sum_pred,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                },
            }
            write_json(scen_dir / "detector_accounts_eval.json", acct_eval)
            counts = acct_eval.get("counts") or acct_eval.get("aggregate") or {}

            logger.info(
                "RQ1 accounts | source=%s | P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)",
                acct_eval.get("predicted_source") or acct_eval.get("source") or "unknown",
                float(counts.get("precision", 0.0)),
                float(counts.get("recall", 0.0)),
                float(counts.get("f1", 0.0)),
                int(counts.get("tp", 0)),
                int(counts.get("fp", 0)),
                int(counts.get("fn", 0)),
            )

        # ---------- RQ2: QA eval ----------
        qa_eval_root = scen_dir / "qa_eval"
        qa_suite_id = "qa"  # will write qa_eval/qa/summary.csv

        qa_summary_csv = qa_eval_root / qa_suite_id / "summary.csv"

        if not args.skip_qa_eval:
            qa_eval_root.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(args.qa_eval_script)),
                "--facts",
                str(suite_jsonl),
                "--semantic-layer-path",
                str(Path(args.semantic_layer_path)),
                "--prompts-dir",
                str(Path(args.prompts_dir)),
                "--outputs-root",
                str(qa_eval_root),
                "--suite-id",
                qa_suite_id,
                "--mode",
                args.mode,
                "--csv-path",
                csv_path,
            ]
            if args.mock_planner:
                cmd.append("--mock-planner")
            if args.mock_analyst:
                cmd.append("--mock-analyst")
            logger.info("Running QA eval subprocess for %s ...", event_id)
            logger.info("QA eval cmd: %s", " ".join(cmd))
            run_cmd(cmd, cwd=repo_root, extra_pythonpath=extra_py)

        if not qa_summary_csv.exists():
            raise FileNotFoundError(f"QA summary.csv not found after eval: {qa_summary_csv}")
        logger.info("QA eval done for %s. Reading %s", event_id, qa_summary_csv)
        qa_metrics = load_qa_metrics(qa_summary_csv)

        # ---------- RQ3: integration ----------
        integ = compute_integration_metrics(
            qa_summary_csv=qa_summary_csv,
            flagged_days_in_window=det_metrics.flagged_days_in_window,
        )
        logger.info(
            "RQ3 integration | detector_injected_rate=%.2f | mention_flagged_day_rate=%.2f | checked=%d",
            integ.detector_context_injected_rate,
            integ.mention_flagged_day_rate,
            integ.n_checked_answers,
        )

        # Scenario-level metrics bundle
        scenario_metrics = {
            "event_id": event_id,
            "scenario_id": scenario_id,
            "anomaly_type": anomaly_type,
            "label_window": {"start": start.isoformat(), "end": end.isoformat()},
            # RQ1
            "rq1": det_metrics.__dict__,
            "rq1_accounts": acct_eval or None,
            # RQ2 (QA)
            "rq2": qa_metrics.__dict__,
            # RQ3 (integration)
            "rq3": integ.__dict__,
        }
        write_json(scen_dir / "scenario_metrics.json", scenario_metrics)
        logger.info("Wrote scenario_metrics.json for %s", event_id)

        # Flatten one row for suite-level CSV
        bundle_rows.append(
            {
                "event_id": event_id,
                "scenario_id": scenario_id,
                "anomaly_type": anomaly_type,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                # RQ1
                "det_hit": det_metrics.hit,
                "det_delay_days": det_metrics.delay_days,
                "det_n_flagged_in_window": det_metrics.n_flagged_in_window,
                "det_n_flagged_nearby_outside": det_metrics.n_flagged_nearby_outside_window,
                "det_top_driver_in_window": det_metrics.top_driver_in_window,
                # RQ1 account attribution (optional)
                "acct_eval_day": (acct_eval.get("comparison_day") if acct_eval else None),
                "acct_pred_source": (acct_eval.get("predicted_source") if acct_eval else None),
                "acct_precision": (
                    acct_eval.get("counts", {}).get("precision") if acct_eval else None
                ),
                "acct_recall": (acct_eval.get("counts", {}).get("recall") if acct_eval else None),
                "acct_f1": (acct_eval.get("counts", {}).get("f1") if acct_eval else None),
                "acct_tp": (acct_eval.get("counts", {}).get("tp") if acct_eval else None),
                "acct_fp": (acct_eval.get("counts", {}).get("fp") if acct_eval else None),
                "acct_fn": (acct_eval.get("counts", {}).get("fn") if acct_eval else None),
                "acct_n_true": (acct_eval.get("counts", {}).get("n_true") if acct_eval else None),
                "acct_n_pred": (acct_eval.get("counts", {}).get("n_pred") if acct_eval else None),
                # RQ2
                "qa_pos_em_rate": qa_metrics.positives_em_rate,
                "qa_neg_refusal_rate": qa_metrics.negatives_refusal_rate,
                "qa_planner_time_window_ok_rate": qa_metrics.planner_time_window_ok_rate,
                "qa_planner_passed_rate": qa_metrics.planner_passed_rate,
                "qa_n_positive_runs": qa_metrics.n_positive_runs,
                "qa_n_negative_runs": qa_metrics.n_negative_runs,
                # RQ3
                "rq3_detector_injected_rate": integ.detector_context_injected_rate,
                "rq3_mention_flagged_day_rate": integ.mention_flagged_day_rate,
                "rq3_n_checked_answers": integ.n_checked_answers,
                # paths
                "scenario_dir": str(scen_dir),
                "qa_summary_csv": qa_metrics.qa_summary_csv,
            }
        )

    # Suite-level outputs
    out_csv = Path(args.out_csv) if args.out_csv else (suite_root / "bundle_results.csv")
    out_jsonl = out_csv.with_suffix(".jsonl")

    df = pd.DataFrame(bundle_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_jsonl(out_jsonl, bundle_rows)

    print(f"Wrote suite results: {out_csv}")
    print(f"Wrote suite results JSONL: {out_jsonl}")


if __name__ == "__main__":
    main()
