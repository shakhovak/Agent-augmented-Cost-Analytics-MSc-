# Semantic layer

The semantic layer defines what the system is allowed to answer in ad‑hoc Q&A mode.
It is a *contract* between the Planner (LLM) and the Executor (deterministic backend).

## What problem it solves

- Prevents unsupported queries (“LLM invents a metric / column”).
- Constrains scope and cost (row caps, time window caps, top‑N caps).
- Makes planning and execution reproducible: the same YAML + same dataset → same query semantics.

## File location

- `configs/semantic_layer.yaml`

## Structure

### `dataset`
Metadata about the source dataset (name, grain, primary keys) and parsing conventions.

### `business_rules`
Domain constants needed for derived KPIs (currency, daily price per account, cost cap).

### `limits`
Safety caps enforced by the executor:

- `max_days_hard`, `max_rows_hard`: hard stops (never exceed).
- `max_days_default`, `max_rows_default`: default planner targets.
- `default_top_n`: used when user asks for “top accounts/services” without a number.
- `drilldown_requires_filters`: prevents high-cardinality scans (e.g., chat_id without account_id).
- `max_group_by_cols`: keeps queries simple and cheap.

### `dimensions`
Allowed grouping keys (date, account_id, chat_type, chat_id).

### `measures`
Raw numeric columns from the dataset (cost components).

### `virtual_dimensions`
Computed “dimensions” that don’t exist as a physical column but can be produced deterministically.
Example: `service_component` unpivots cost columns into (component, value).

### `derived_metrics`
Metrics computed by the executor from measures and/or business rules.
Important: the LLM must **not** compute new numbers; it only requests these metrics.

### `allowed_operations`
Enumerates allowed aggregations, filters, group_by keys, sorting, and chart types.

## Examples

### Supported
- “Top 10 accounts by total cost services yesterday”
- “Trend of total cost services over last 14 days”
- “Cost breakdown by service component for a given day”
- “Compare WhatsApp vs Telegram costs last 7 days” (chat_type)

### Unsupported (should be rejected by planner validation)
- “Forecast next week’s cost” (no forecasting in semantic layer)
- “Breakdown by country” (dimension not defined)
- “Show raw messages / prompts per user” (not in dataset)

## How limits are enforced

1. Planner proposes a plan referencing *only* allowed dimensions/metrics/filters.
2. Executor validates the plan against this YAML.
3. Executor clamps / rejects requests exceeding hard caps (days, rows, top‑N).
4. Evidence Pack is generated from executed QuerySpecs; Analyst/verifier use only Evidence Pack.
