#!/usr/bin/env python3

"""
Generate negative (UNSUPPORTED) evaluation cases as JSONL.

Produces records like:
{
  "id": 9001,
  "type": "negative_case",
  "category": "relative_date",
  "expected": {"planner_status": "UNSUPPORTED", "reason_contains": ["relative", "explicit date"]},
  "question_variants": [...5 questions...],
  "anchor_day": "2025-12-31",
  "csv_fingerprint": {"csv_path": "...", "anchor_day": "2025-12-31", "seed": 123}
}

Usage:
  python generate_negative_cases.py \
    --out outputs/facts/negative_cases.jsonl \
    --csv-path data/joint_info_2025_with_anomalies.csv \
    --anchor-day 2025-12-31 \
    --seed 123 \
    --start-id 9001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _variants_relative_date() -> list[str]:
    # Intentionally violates MVP limitation (relative dates). Expect UNSUPPORTED.
    return [
        "What were the total costs yesterday?",
        "Budget check — how much did we spend today in total?",
        "Can you show overall spend for last week?",
        "What is our total cost this month so far?",
        "How much have we spent recently (overall total cost)?",
    ]


def _variants_out_of_year() -> list[str]:
    # Explicit non-2025 dates. Expect UNSUPPORTED due to dataset coverage.
    return [
        "What is the total cost for 2026-01-10?",
        "How much did we spend on 10/01/2026 in total?",
        "Give the aggregate cost for December 31, 2024.",
        "Total cost on 2024-12-31?",
        "What was overall spend on 1 Feb 2026?",
    ]


def _variants_unsupported_dimension() -> list[str]:
    # Keeps a valid 2025 date but asks for a dimension not in semantic layer.
    return [
        "Break down total cost by region for 2025-01-20.",
        "Show total spend by country on 20/01/2025.",
        "For January 20, 2025, split costs by manager.",
        "Give total cost by customer segment for 20 Jan 2025.",
        "For 2025-01-20, show costs by plan tier.",
    ]


def _variants_unsupported_metric() -> list[str]:
    # Valid 2025 date but metric not supported (revenue/ROI/etc.)
    return [
        "What is the revenue for 2025-01-20?",
        "Show ROI for 20/01/2025.",
        "What was profit on January 20, 2025?",
        "Give LTV for 20 Jan 2025.",
        "What is CAC for 2025-01-20?",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate negative UNSUPPORTED cases (JSONL).")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--csv-path",
        required=True,
        help="CSV path to include in csv_fingerprint for evaluator",
    )
    ap.add_argument("--anchor-day", default="2025-12-31")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--start-id", type=int, default=9001)
    args = ap.parse_args()

    out_path = Path(args.out)

    records: list[dict[str, Any]] = []
    next_id = args.start_id

    def add_case(category: str, variants: list[str], reason_contains: list[str]) -> None:
        nonlocal next_id
        records.append(
            {
                "id": next_id,
                "type": "negative_case",
                "category": category,
                "expected": {
                    "planner_status": "UNSUPPORTED",
                    # Keep this loose; you can remove reason_contains entirely if you want status-only scoring
                    "reason_contains": reason_contains,
                },
                "question_variants": variants,
                "anchor_day": args.anchor_day,
                "csv_fingerprint": {
                    "csv_path": args.csv_path,
                    "anchor_day": args.anchor_day,
                    "seed": args.seed,
                },
            }
        )
        next_id += 1

    add_case(
        "relative_date",
        _variants_relative_date(),
        reason_contains=["relative", "explicit", "date", "2025"],
    )
    add_case(
        "out_of_year",
        _variants_out_of_year(),
        reason_contains=["2025", "only", "data"],
    )
    add_case(
        "unsupported_dimension",
        _variants_unsupported_dimension(),
        reason_contains=["unsupported", "dimension"],
    )
    add_case(
        "unsupported_metric",
        _variants_unsupported_metric(),
        reason_contains=["unsupported", "metric"],
    )

    _write_jsonl(out_path, records)
    print(f"Wrote {len(records)} negative cases to {out_path}")


if __name__ == "__main__":
    main()
