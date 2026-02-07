# Evidence Pack Contract v0.1

## Purpose
The Evidence Pack is the deterministic interface between the analytics layer and the agent/explainer layer.
It contains small, structured tables with stable schemas. The agent must **not** compute KPIs; it must
only narrate and reason over the Evidence Pack contents.

This contract defines:
- required table keys
- required columns, types, and semantics
- high-level computation rules

### Conventions
- All tables are `pandas.DataFrame` objects, returned in a dict keyed by table name.
- `date` is a calendar day (UTC-normalized), represented as `datetime.date` in code.
- Currency units: **cost fields are in platform billing currency units** (synthetic), consistent across tables.
- Percent deltas are fractions (e.g., `0.10` = +10%), not percentages.

### Versioning
Breaking changes (rename/remove required keys or columns, change semantics) require a **major** version bump.
Additive changes (new optional columns/tables) require a **minor** version bump.

---

## Required tables (v0.1)

### 1) `kpis` — single-row summary (report day vs previous day)
**Shape:** exactly 1 row.

**Primary meaning:** headline metrics for the report day with previous-day values and deltas, suitable for
narrative summaries and anomaly explanations.

**Required columns**
| column | type | meaning |
|---|---|---|
| `date` | date | report day |
| `total_cost` | float | sum of `total_cost` for report day |
| `total_cost_prev` | float | sum of `total_cost` for previous day |
| `total_cost_delta_abs` | float | `total_cost - total_cost_prev` |
| `total_cost_delta_pct` | float | `(total_cost - total_cost_prev) / total_cost_prev` with safe divide-by-zero handling |

**Recommended (present in v0.1 builder; may expand over time)**
The KPI table may include additional metrics following the same suffix pattern:
- `<metric>`: current value for report day
- `<metric>_prev`: previous-day value
- `<metric>_delta_abs`: absolute delta
- `<metric>_delta_pct`: percent delta (fraction)

Common metrics included:
- `active_users` (float): number of accounts with `total_cost > 0` on that day
- `active_account_days` (float): distinct `(account_id, date)` with `total_cost > 0`
- `dials_analyzed` (float): unique `(account_id, chat_id, chat_type)` among rows with `total_cost > 0`
- `avg_cost_per_account` (float): `total_cost / active_users` (0 if no active users)
- `avg_cost_per_active_account_day` (float): `total_cost / active_account_days` (0 if none)
- `avg_cost_per_active_service_account` (float): mean spend across service-active cohorts (see KPI definitions)

Row-density / cap diagnostics (if cap is configured) may also be included:
- `rows_per_account_day_p95`, `rows_per_account_day_max`, `rows_per_account_day_n_over_cap`, `rows_per_account_day_pct_over_cap`
(with `_prev`/`_delta_*` variants).

**High-level computation**
- Filter `df_all` to `date == report_day` for current values.
- Filter `df_all` to `date == report_day - 1 day` for previous values.
- Compute metrics deterministically using KPI definitions from `analytics/kpi_definitions.py`.

**Example snippet**
| date | total_cost | total_cost_prev | total_cost_delta_abs | total_cost_delta_pct |
|---|---:|---:|---:|---:|
| 2025-01-15 | 25.66 | 19.71 | 5.95 | 0.30 |

---

### 2) `trend_daily` — recent daily trend table
**Shape:** up to `trend_days` rows (inclusive of report day), sorted ascending by date.

**Primary meaning:** compact time-series evidence for short-term trend/context used in explanations.

**Required columns**
| column | type | meaning |
|---|---|---|
| `date` | date | calendar day |
| `total_cost` | float | daily sum of `total_cost` |
| `active_users` | int | daily count of active users/accounts |
| `active_account_days` | int | daily active account-days |
| `dials_analyzed` | int | daily unique (account_id, chat_id, chat_type) with cost>0 |

**High-level computation**
- Filter `df_all` to the date range `[report_day - (trend_days-1), report_day]`.
- Group by `date` to compute daily aggregates.

**Example snippet**
| date | total_cost | active_users | active_account_days | dials_analyzed |
|---|---:|---:|---:|---:|
| 2025-01-13 | 33.49 | 3 | 3 | 3 |
| 2025-01-14 | 19.71 | 1 | 1 | 1 |
| 2025-01-15 | 25.66 | 1 | 1 | 1 |

---

### 3) `service_breakdown` — cost contribution by top-level component (report day)
**Shape:** 4 rows in v0.1 (top-level components), sorted by `cost` descending.

**Primary meaning:** identifies *which service/component* drove spend for the report day.

**Top-level components (v0.1)**
- `cost_dialog`
- `total_cost_tasks`
- `total_cost_classifications`
- `cost_amocrm_call`

**Required columns**
| column | type | meaning |
|---|---|---|
| `date` | date | report day |
| `component` | str | top-level component name |
| `cost` | float | sum of component cost for report day |
| `share_of_total` | float | `cost / sum(top-level component costs)` |

**Notes**
- `service_breakdown` intentionally uses **top-level components only** to avoid double counting.
  Detailed subcomponents (e.g., `cost_qc`, `cost_check_list`) may appear in optional tables.

**Example snippet**
| date | component | cost | share_of_total |
|---|---|---:|---:|
| 2025-01-15 | total_cost_classifications | 25.66 | 1.00 |
| 2025-01-15 | cost_dialog | 0.00 | 0.00 |

---

### 4) `top_accounts` — top-N accounts by cost (report day)
**Shape:** up to `top_n` rows, sorted by `total_cost` descending.

**Primary meaning:** highlights which accounts contributed most to spend and supports drill-down.

**Required columns**
| column | type | meaning |
|---|---|---|
| `date` | date | report day |
| `rank` | int | 1..N (dense rank) |
| `account_id` | int/str | account identifier |
| `total_cost` | float | per-account sum of `total_cost` for report day |
| `share_of_total_cost` | float | `total_cost / (sum total_cost for report day)` |

**Example snippet**
| date | rank | account_id | total_cost | share_of_total_cost |
|---|---:|---:|---:|---:|
| 2025-01-15 | 1 | 21782287 | 25.66 | 1.00 |

---

## Optional tables (non-breaking extras)
The builder may also return additional tables for convenience. These are **not required** by v0.1
and may change without breaking the contract.

Typical optional keys:
- `service_totals`: component totals including subcomponents (debug/detail use)
- `distribution_hist`: histogram table for a metric (e.g., per-account costs)
- `distribution_stats`: summary stats for a metric distribution
- `exceptions_queue`: placeholder for rule-based exceptions (later milestones)

---

## Error / edge-case behavior
- If `report_day` has **no rows** in the input dataset, the builder raises a `ValidationError`.
- If the previous day has no rows, previous KPI values evaluate to 0 deterministically.
- If total cost is 0, share columns are set to 0 to avoid NaNs.

---

## Provenance
This contract corresponds to the Evidence Pack builder implemented in:
- `src/cost_agent_mvp/analytics/evidence_pack.py`
and KPI definitions in:
- `src/cost_agent_mvp/analytics/kpi_definitions.py`
