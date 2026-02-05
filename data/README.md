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
- `joint_info_2025.csv` — generated output (usually git-ignored)
- `samples/` — optional small outputs for quick inspection

> Tip: keep generated CSVs out of version control. Commit only configs + code + a tiny sample if needed.

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
- `joint_info_2025.csv` — generated output (usually git-ignored)
- `samples/` — optional small outputs for quick inspection

> Tip: keep generated CSVs out of version control. Commit only configs + code + a tiny sample if needed.

---

## Quickstart

Run from the **repo root**.

### 1) Generate a small sample (fast)

```bash
python -m data.generator --config data/samples/sample_config.yaml
````

### 2) Generate the full dataset (slower)

```bash
python -m data.generator --config data/config_data_generator.yaml
```

The output path is controlled by `output.path` in the config.
By default it writes something like:

* `data/joint_info_2025.csv`

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

## Sanity checks (recommended)

After generation, run a few quick checks in Python:

```python
import pandas as pd

df = pd.read_csv("data/joint_info_2025.csv")

print("shape:", df.shape)
print("date range:", df["date"].min(), "→", df["date"].max())
print("num_services distribution:")
print(df["num_services"].value_counts(normalize=True).sort_index())
```

### “New clients per day” (correct definition)

When plotting “daily new clients”, define **new client** as:

> an account whose **first ever** service use date is that day

Example for one service:

```python
import pandas as pd

df = pd.read_csv("data/joint_info_2025.csv")
df["date"] = pd.to_datetime(df["date"])

feature_col = "has_tasks"  # or has_classifications / has_amocrm_call
feature_df = df[df[feature_col] == 1]

first_use = feature_df.groupby("account_id")["date"].min()
daily_new = first_use.value_counts().sort_index()

print("total adopters:", first_use.shape[0])
print("sum daily new:", daily_new.sum())
```

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

**Practical tuning workflow:**

1. generate sample config (fast)
2. inspect:

   * adopters totals per service
   * service-days counts
   * `num_services` distribution
3. adjust 1–2 knobs, rerun
4. only then run full config

---

## Ethics / data handling note

* The dissertation uses **only synthetic data** (no PII, no customer content, no organisational logs).
* It is safe to publish:

  * generator code
  * YAML configs
  * scenario definitions / evaluation scripts
  * small sample synthetic datasets
* Do **not** commit large generated outputs unless explicitly intended.

---

## Troubleshooting

### YAML parse errors

If you see errors like:

* `yaml.scanner.ScannerError: mapping values are not allowed here`

It usually means indentation or `:` formatting. Recheck the line number printed by the error.

### KeyError: missing config section

Example:

* `KeyError: 'services_usage_days'`

Your config is missing a required top-level block. Compare against `config_data_generator.yaml`.

### `UnboundLocalError: cannot access local variable 'np'`

Make sure the module has `import numpy as np` at the top and you didn’t shadow `np` inside a function.

### Generation is slow

* Use the sample config first.
* Ensure tqdm progress bars are enabled where loops are large.
* If needed, reduce `accounts.n_accounts` or shorten the date range temporarily.

---

## Command reference

```bash
# Generate sample dataset
python -m data.generator --config data/samples/sample_config.yaml

# Generate full dataset
python -m data.generator --config data/config_data_generator.yaml
```

```
```
