# Data folder — synthetic dataset generation

This folder contains the **synthetic data generator** used in the dissertation.
It produces a joint “usage-cost” dataset that mimics a subscription AI platform’s operational metadata (accounts, chats, daily activity, service usage, and costs) **without using any real organisational data**.

The generator is designed to be:
- **Reproducible** (same config + seed → identical output)
- **Config-driven** (all modelling assumptions live in YAML)
- **Compatible with the analytics pipeline** (schema/type coercion + required columns)

---

## Folder layout

Typical contents:

- `config_data_generator.yaml` — **main configuration** (single source of truth; used as Appendix in the dissertation)
- `samples/sample_config.yaml` — small config for quick local runs and tests
- `generator.py` — CLI entry point for dataset generation
- `service_costs.py` — service adoption/retention schedules + usage calendar + cost assignment logic
- `users_activity.py` — analysis helpers / sanity-check scripts (plots, growth/retention/activity summaries)
- `joint_info_2025.csv` — generated output (git-ignored)
- `samples/` — optional small outputs for quick inspection

---

## Quickstart

Run from the **repo root**.
### 1) Generate a small sample (fast)

```bash
python -m data.generator --config data/samples/sample_config.yaml
```

### 2) Generate the full dataset (slower)
```bash
python -m data.generator --config data/config_data_generator.yaml
```

The output path is controlled by output.path in the config.

CLI arguments can override YAML for quick experiments. YAML remains the authoritative source of modelling assumptions used in the dissertation.

---

## What the generator produces

The main output is a **CSV “joint table”** with one row per **active chat-day**:

* identifiers: `account_id`, `chat_id`, `chat_type`, `date`
* service flags (row-level): `has_tasks`, `has_classifications`, `has_amocrm_call`
* service mix: `num_services`, `has_both`
* costs: `cost_dialog`, `total_cost_tasks`, `total_cost_classifications`, `cost_amocrm_call`, `total_cost`
* optional volume counters (if enabled in code): `n_tasks`, `n_classifications`, `n_amocrm_calls`

**Important modelling note:**
By default the generator assigns each account-day’s service usage to **one chosen active chat row** (`services_model.distribute_to_chats.mode: one_chat`).
This keeps the table compact and prevents accidental double-counting when aggregating by account-day.

---

## Output schema and validation

The analytics pipeline expects a strict minimal schema:

* required columns must exist
* dates must parse reliably
* numeric columns must coerce safely (bad non-null values should fail fast)

Schema/type coercion lives in:

* `src/cost_agent_mvp/data/schema.py`

Typical usage pattern in code:

* `validate_columns(df, default_joint_schema())`
* `coerce_types(df, default_joint_schema())`

---

## Reproducibility / determinism

Determinism is controlled by:

* `dataset.seed`

Rules:

* Same YAML + same seed → **identical output** (stable for experiments)
* Change seed → a **different** synthetic dataset instance with the same statistical behaviour

This is critical for:

* repeating evaluations
* comparing model versions fairly
* anomaly injection experiments (later steps)

---

## How to tune the generator (main knobs)

Most tuning happens in `config_data_generator.yaml`.

| Goal                                         | Primary knobs in YAML                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------------------------- |
| More overall activity (more service-days)    | `services_usage_days.mean_active_days`, `services_model.p_convert_after_trial`           |
| Longer retention windows                     | `services_model.retention_targets`, `services_model.post90_daily_churn`                  |
| More new adopters                            | `services_model.new_adopters_per_month`                                                  |
| Stronger “first half of month” adoption bias | `services_model.adoption_day_bias.alpha`                                                 |
| Stronger weekend/summer dips                 | `activity.dow_multiplier`, `activity.month_multiplier`                                   |
| More daily volume (events/day)               | `services_event_volume.mean_events_per_active_day`, `services_event_volume.dispersion_k` |
| Heavier-tailed costs / more variance         | `services.cost_when_used.*.sigma`, `services.cost_when_used.*.max`                       |
| More multi-service days (bundling)           | planned next step: add service correlation rules (not enabled by default)                |

---

## Anomaly injection (M1-06)

This project supports injecting controlled anomalies into the synthetic dataset and producing **ground-truth labels** (JSONL) for evaluation.

### Outputs

Running the injector produces:

- `data/sample_with_anomalies.csv` — synthetic dataset with injected anomalies
- `data/labels.jsonl` — ground-truth anomaly events (one JSON object per line)

Large generated outputs should not be committed to git. You may commit a small sample for demos/tests.

### 1) Generate base synthetic dataset

Generate the base dataset (normal behavior) first:

```powershell
python data/generator.py --config config_data_generator.yaml --output data/sample.csv

```
If you run the generator without CLI overrides, it will follow the config file defaults (e.g., full-year 2025, `n_accounts=3000`).

### 2) Inject anomalies + produce labels

Inject anomalies defined in `configs/anomaly_scenarios.yaml`:

```powershell
python scripts/inject_anomalies.py `
  --input data/sample.csv `
  --config configs/anomaly_scenarios.yaml `
  --output data/sample_with_anomalies.csv `
  --labels data/labels.jsonl
```

The injection is **reproducible**: given the same base dataset + scenario config + seeds, the outputs are deterministic.

### 3) Scenario configuration

Anomaly scenarios are stored in:

* `configs/anomaly_scenarios.yaml`

Each scenario defines:

* `scenario_id`, `anomaly_type`
* `seed` (controls deterministic account selection / row updates)
* `account_selection` (e.g., `random_k`)
* `window` (`start_date`, `duration_days`)
* `params` (type-specific parameters like `magnitude` or `exceed_by`)
* `driver_template` (human-readable explanation template)

Supported anomaly types (v0):

* `cost_spike` — unit-cost spike (e.g., dialog/task component cost multiplier)
* `volume_spike` — usage spike (e.g., `n_tasks` increases and cost scales)
* `cap_spike` — concurrent chat cap violation (adds extra chat rows per account-day)

### 4) Labels format (JSONL)

`data/labels.jsonl` contains one JSON object per anomaly event with (minimum) fields:

* `event_id`
* `scenario_id`
* `seed`
* `anomaly_type`
* `affected_services` (JSON list)
* `start_date`, `end_date` (ISO dates)
* `driver_template`
