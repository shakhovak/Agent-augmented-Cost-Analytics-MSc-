from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# ---------------------------
# IO helpers
# ---------------------------


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


def iso_to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def date_to_iso(d: date) -> str:
    return d.isoformat()


def daterange(start: date, end: date) -> list[date]:
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]


def run_cmd(cmd: list[str], *, cwd: Path, extra_pythonpath: list[Path] | None = None) -> None:
    env = os.environ.copy()
    add_paths = [str(p) for p in (extra_pythonpath or []) if p.exists()]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(add_paths + ([existing] if existing else []))

    p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  cwd: {cwd}\n"
            f"  stdout:\n{p.stdout}\n"
            f"  stderr:\n{p.stderr}\n"
        )


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


# ---------------------------
# Core compare logic (event-aware version)
# ---------------------------


def _true_accounts_for_day(
    label_event: dict[str, Any],
    compare_day: str,
    *,
    label_by_day_field: str,
) -> set[int]:
    true_accounts: set[int] = set()

    by_day = label_event.get(label_by_day_field, {})
    if isinstance(by_day, dict):
        xs = by_day.get(compare_day, [])
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
    det: dict[str, Any],
    *,
    anomaly_type: str,
    use_top_accounts_if_missing: bool,
) -> tuple[set[int], str | None]:
    decision = det.get("decision", {}) if isinstance(det.get("decision", {}), dict) else {}

    pred_accounts: set[int] = set()
    pred_source: str | None = None

    if isinstance(decision.get("flagged_account_ids"), list):
        pred_accounts = {int(x) for x in decision["flagged_account_ids"]}
        pred_source = "flagged_account_ids"

    elif isinstance(decision.get("account_decisions"), list):
        flag_key = f"flag_{anomaly_type}"
        for row in decision["account_decisions"]:
            if not isinstance(row, dict):
                continue
            if row.get(flag_key) is True and "account_id" in row:
                pred_accounts.add(int(row["account_id"]))
        pred_source = "account_decisions"

    else:
        if not use_top_accounts_if_missing:
            raise ValueError(
                "Detection JSON does not contain decision.flagged_account_ids or decision.account_decisions "
                "and use_top_accounts_if_missing=False."
            )
        top = decision.get("top_accounts", []) or []
        for row in top:
            if isinstance(row, dict) and "account_id" in row:
                pred_accounts.add(int(row["account_id"]))
        pred_source = "top_accounts"

    return pred_accounts, pred_source


def compare_label_event_vs_detection(
    *,
    label_event: dict[str, Any],
    detection_json_path: str,
    anomaly_type: str,
    label_by_day_field: str,
    day: str | None = None,
    use_top_accounts_if_missing: bool = True,
    max_list: int = 200,
) -> dict[str, Any]:
    with open(detection_json_path, encoding="utf-8") as f:
        det = json.load(f)

    decision = det.get("decision", {}) if isinstance(det.get("decision", {}), dict) else {}
    compare_day = day or decision.get("day") or label_event.get("start_date")

    true_accounts = _true_accounts_for_day(
        label_event, compare_day, label_by_day_field=label_by_day_field
    )
    pred_accounts, pred_source = _pred_accounts_from_detection(
        det,
        anomaly_type=anomaly_type,
        use_top_accounts_if_missing=use_top_accounts_if_missing,
    )

    tp = pred_accounts & true_accounts
    fp = pred_accounts - true_accounts
    fn = true_accounts - pred_accounts

    precision = (len(tp) / len(pred_accounts)) if pred_accounts else 0.0
    recall = (len(tp) / len(true_accounts)) if true_accounts else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    def _clip(xs: set[int]) -> list[int]:
        out = sorted(xs)
        return out[:max_list]

    out: dict[str, Any] = {
        "anomaly_type": anomaly_type,
        "label_event_id": label_event.get("event_id"),
        "label_window": {
            "start_date": label_event.get("start_date"),
            "end_date": label_event.get("end_date"),
        },
        "comparison_day": compare_day,
        "label_by_day_field": label_by_day_field,
        "predicted_source": pred_source,
        "counts": {
            "n_true": len(true_accounts),
            "n_pred": len(pred_accounts),
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "tp_accounts": _clip(tp),
        "fp_accounts": _clip(fp),
        "fn_accounts": _clip(fn),
    }

    if pred_source == "top_accounts":
        out["note"] = (
            "Predicted accounts derived from decision.top_accounts only (top_k subset). "
            "Prefer decision.flagged_account_ids or decision.account_decisions for full-set evaluation."
        )

    return out


# ---------------------------
# Evaluation driver
# ---------------------------


@dataclass
class EventSummary:
    event_id: str
    anomaly_type: str
    start_date: str
    end_date: str
    day_mode: str
    n_days_evaluated: int
    # aggregate metrics
    precision_mean: float
    recall_mean: float
    f1_mean: float
    precision_best: float
    recall_best: float
    f1_best: float


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate detector account-level precision/recall vs labels.jsonl."
    )
    ap.add_argument("--labels", required=True, help="Path to labels.jsonl")
    ap.add_argument("--csv", required=True, help="Injected CSV path (used by detector)")
    ap.add_argument(
        "--out-root",
        default="outputs/anomaly_eval",
        help="Where to write evaluation outputs",
    )
    ap.add_argument(
        "--detector-script",
        default="src/cost_agent_mvp/cli/run_detector_7d.py",
        help="Detector CLI script path",
    )
    ap.add_argument("--label-by-day-field", default="requested_accounts_by_day")
    ap.add_argument(
        "--anomaly-type",
        default=None,
        help="Filter labels by anomaly_type (e.g., cost_spike)",
    )
    ap.add_argument("--event-id", default=None, help="Evaluate only this label event_id")
    ap.add_argument(
        "--day-mode",
        choices=["start", "end", "each"],
        default="start",
        help="Which day(s) in the label window to evaluate",
    )
    ap.add_argument("--use-top-accounts-if-missing", action="store_true", default=True)
    ap.add_argument(
        "--max-list",
        type=int,
        default=100,
        help="Max account ids to store in tp/fp/fn lists",
    )

    ap.add_argument(
        "--scenario-suite-root",
        default=None,
        help="If provided, also write per-event eval into <suite_root>/<event_id>/detector_accounts_eval.json",
    )

    # Pass-through args to the detector CLI (thresholds etc.)
    ap.add_argument(
        "--detector-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed through to run_detector_7d.py (e.g., --z-threshold 4.5). "
        "Usage: --detector-args --arg1 val1 --arg2 val2",
    )

    ap.add_argument("--limit", type=int, default=None, help="Limit number of events for debugging")
    args = ap.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    extra_py = [repo_root, repo_root / "src"]

    labels = read_jsonl(Path(args.labels))

    # filter
    if args.anomaly_type:
        labels = [ev for ev in labels if ev.get("anomaly_type") == args.anomaly_type]
    if args.event_id:
        labels = [ev for ev in labels if ev.get("event_id") == args.event_id]
    if args.limit is not None:
        labels = labels[: args.limit]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    day_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    suite_root = Path(args.scenario_suite_root) if args.scenario_suite_root else None

    for ev in tqdm(labels, desc="Label events", unit="event"):
        event_id = str(ev.get("event_id"))
        anomaly_type = str(ev.get("anomaly_type") or "unknown")
        start = iso_to_date(str(ev.get("start_date")))
        end = iso_to_date(str(ev.get("end_date")))

        if args.day_mode == "start":
            days = [start]
        elif args.day_mode == "end":
            days = [end]
        else:
            days = daterange(start, end)

        per_day_metrics: list[tuple[float, float, float]] = []

        # per-event output dirs
        ev_dir = out_root / event_id
        ev_dir.mkdir(parents=True, exist_ok=True)

        for d in days:
            d_iso = d.isoformat()
            det_out = ev_dir / f"detector_{d_iso}.json"

            # 1) run detector for this end-day
            cmd = [
                sys.executable,
                str(Path(args.detector_script)),
                "--csv",
                str(args.csv),
                "--end-day",
                d_iso,
                "--out",
                str(det_out),
                "--include-account-decision",
            ]
            # pass-through extra args
            if args.detector_args:
                cmd += args.detector_args

            run_cmd(cmd, cwd=repo_root, extra_pythonpath=extra_py)

            # 2) evaluate accounts vs label for this day
            comp = compare_label_event_vs_detection(
                label_event=ev,
                detection_json_path=str(det_out),
                anomaly_type=anomaly_type,
                label_by_day_field=args.label_by_day_field,
                day=d_iso,
                use_top_accounts_if_missing=args.use_top_accounts_if_missing,
                max_list=args.max_list,
            )

            counts = comp["counts"]
            per_day_metrics.append((counts["precision"], counts["recall"], counts["f1"]))

            day_rows.append(
                {
                    "event_id": event_id,
                    "anomaly_type": anomaly_type,
                    "label_start": date_to_iso(start),
                    "label_end": date_to_iso(end),
                    "day": d_iso,
                    "predicted_source": comp.get("predicted_source"),
                    "n_true": counts["n_true"],
                    "n_pred": counts["n_pred"],
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "precision": counts["precision"],
                    "recall": counts["recall"],
                    "f1": counts["f1"],
                    "detector_json": str(det_out),
                }
            )

            # store per-day comparison json for audit/debug
            write_json(ev_dir / f"compare_{d_iso}.json", comp)

        # per-event aggregation
        if per_day_metrics:
            p = [x[0] for x in per_day_metrics]
            r = [x[1] for x in per_day_metrics]
            f = [x[2] for x in per_day_metrics]
            summary = EventSummary(
                event_id=event_id,
                anomaly_type=anomaly_type,
                start_date=date_to_iso(start),
                end_date=date_to_iso(end),
                day_mode=args.day_mode,
                n_days_evaluated=len(per_day_metrics),
                precision_mean=float(sum(p) / len(p)),
                recall_mean=float(sum(r) / len(r)),
                f1_mean=float(sum(f) / len(f)),
                precision_best=float(max(p)),
                recall_best=float(max(r)),
                f1_best=float(max(f)),
            )
            event_rows.append(summary.__dict__)

            # store per-event summary
            write_json(ev_dir / "event_summary.json", summary.__dict__)

            # optionally mirror into scenario suite folder
            if suite_root is not None:
                scen_dir = suite_root / event_id
                if scen_dir.exists():
                    write_json(scen_dir / "detector_accounts_eval.json", summary.__dict__)

    # write global outputs
    df_day = pd.DataFrame(day_rows)
    df_event = pd.DataFrame(event_rows)

    df_day.to_csv(out_root / "results_day.csv", index=False)
    df_event.to_csv(out_root / "results_event.csv", index=False)

    print(f"Wrote: {out_root / 'results_day.csv'}")
    print(f"Wrote: {out_root / 'results_event.csv'}")


if __name__ == "__main__":
    main()
