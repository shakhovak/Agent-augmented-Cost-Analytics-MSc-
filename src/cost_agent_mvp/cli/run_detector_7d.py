from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

# Keep this import if you replaced cost_agent_mvp/agent/detector.py with the updated detector.
# If you kept it as a separate module, change the import to: from detector_updated import ...
from cost_agent_mvp.agent.detector import (
    detect_anomaly_for_end_day,
    format_detector_brief,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run account×day anomaly detector (robust 14-day baseline; low-FP defaults)."
    )
    ap.add_argument("--csv", required=True, help="Path to joint daily CSV")
    ap.add_argument("--end-day", required=True, help="Decision day in YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="Write JSON result here")

    # Windows
    ap.add_argument(
        "--baseline-days",
        type=int,
        default=14,
        help="Baseline window length (previous N days, excluding end-day and the last --gap-days). Default: 14",
    )
    ap.add_argument(
        "--context-days",
        type=int,
        default=7,
        help="Context window included in output (last N days ending end-day). Default: 7",
    )

    ap.add_argument(
        "--gap-days",
        type=int,
        default=2,
        help="Exclude the last N days from the baseline window to avoid contamination by multi-day anomalies. Default: 2",
    )

    # Core thresholds
    ap.add_argument(
        "--z-threshold",
        type=float,
        default=5.5,
        help="Robust z threshold for baseline-based spikes. Default: 4.5",
    )
    ap.add_argument(
        "--chat-cap",
        type=int,
        default=70,
        help=(
            "Hard cap for concurrent chats per account-day (>= triggers cap_spike). "
            "Use 0 to disable hard cap. Default: 70"
        ),
    )
    ap.add_argument(
        "--min-baseline-points",
        type=int,
        default=7,
        help="Minimum non-zero baseline points per metric to trust baseline stats. Default: 7",
    )

    # Materiality gates (big FP reducer)
    ap.add_argument(
        "--min-account-impact-services",
        type=float,
        default=20.0,
        help="Minimum excess services cost to flag an account for cost/volume spikes. Default: 10.0",
    )
    ap.add_argument(
        "--min-account-impact-dialog",
        type=float,
        default=1.0,
        help="Minimum excess dialog cost to flag an account for dialog-driven cost spikes. Default: 1.0",
    )

    # Ratio gates (recommended for multiplier injections)
    ap.add_argument(
        "--ratio-cost",
        type=float,
        default=10.0,
        help="Minimum cost ratio vs baseline median to consider cost_spike. Default: 5.0",
    )
    ap.add_argument(
        "--ratio-volume",
        type=float,
        default=5.0,
        help="Minimum volume ratio vs baseline median to consider volume_spike. Default: 5.0",
    )

    # Cap spike baseline jump gate (used when hard cap is disabled or alongside it)
    ap.add_argument(
        "--abs-chat-jump",
        type=int,
        default=10,
        help="Minimum absolute jump in chats vs baseline median to consider cap_spike (baseline-based). Default: 10",
    )

    ap.add_argument(
        "--enable-chat-jump",
        action="store_true",
        help="Also detect cap_spike via baseline jump + z-score even when n_chats is below the hard cap. Default: off (hard-cap only).",
    )

    # Day-level decision
    ap.add_argument(
        "--min-accounts",
        type=int,
        default=1,
        help="Minimum flagged accounts to mark the day anomalous. Default: 1",
    )
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum total excess cost to mark the day anomalous. Default: 0.0",
    )

    # Output shaping
    ap.add_argument(
        "--top-k-accounts",
        type=int,
        default=10,
        help="How many top accounts to include in the evidence section. Default: 10",
    )

    ap.add_argument(
        "--include-account-decisions",
        action="store_true",
        help="Include per-account decision details for ALL accounts on the day (large JSON, useful for FN/FP analysis).",
    )

    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging + include debug block in output JSON (if supported).",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    chat_cap = None if int(args.chat_cap) == 0 else int(args.chat_cap)

    det = detect_anomaly_for_end_day(
        csv_path=args.csv,
        end_day=args.end_day,
        baseline_days=int(args.baseline_days),
        context_days=int(args.context_days),
        gap_days=int(args.gap_days),
        # legacy alias used by older detector versions; still accepted
        z_threshold=float(args.z_threshold),
        chat_cap=chat_cap,
        min_accounts=int(args.min_accounts),
        min_delta=float(args.min_delta),
        min_baseline_points=int(args.min_baseline_points),
        min_account_impact_services=float(args.min_account_impact_services),
        min_account_impact_dialog=float(args.min_account_impact_dialog),
        ratio_cost=float(args.ratio_cost),
        ratio_volume=float(args.ratio_volume),
        abs_chat_jump=int(args.abs_chat_jump),
        enable_chat_jump=bool(args.enable_chat_jump),
        top_k_accounts=int(args.top_k_accounts),
        debug=bool(args.debug),
        include_account_decisions=bool(args.include_account_decisions),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(det, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(format_detector_brief(det))


if __name__ == "__main__":
    main()
