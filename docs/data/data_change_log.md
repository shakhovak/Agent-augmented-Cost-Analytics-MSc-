

# Dataset Changelog

This changelog tracks changes to the synthetic dataset schema, generator parameters, and anomaly scenarios.

Versioning follows Semantic Versioning:
- MAJOR: incompatible schema changes
- MINOR: new fields/scenarios added in backward-compatible way
- PATCH: bugfixes or parameter tweaks that do not change schema

## 0.1.0 — Base synthetic dataset (initial)
- Added base generator producing `data/sample.csv` for 2025.
- Schema includes account/chat identifiers, daily date, component costs, totals, and service flags.
- Added schema validation + type coercion via `load_csv()`.

## 0.1.1 — Anomaly injection + ground truth labels (v0)
- Added anomaly scenarios config: `data/anomaly_scenarios.yaml`.
- Added injector script producing:
  - `data/sample_with_anomalies.csv`
  - `data/labels.jsonl` (ground truth JSONL)
- Supported anomaly types v0:
  - `cost_spike` (component multiplier)
  - `volume_spike` (count and cost scaling)
  - `cap_spike` (concurrency/cap violation via extra rows)

## 0.1.2 — Documentation pass (M1-07)
- Added `data/data_dictionary.csv`.
- Expanded `data/README.md` to document generation, injection, labels schema, and scenarios.
