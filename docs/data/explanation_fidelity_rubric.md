## Explanation Fidelity Rubric

This rubric is for **manual spot‑checks of automated evaluation** of QA explanations over the cost analytics scenarios (facts, positive/negative cases, question variants).

### Scope

- **What is being judged**: the **explanation text** the agent produces in response to a question, not the raw numeric answer alone.
- **Ground truth**: the structured `fact` or `negative_case` record (e.g. from `facts.jsonl`, `positives_with_questions.jsonl`, `negatives.jsonl`) including:
  - `type` (e.g. `total_cost_single_day`, `top_service_by_cost_single_day`, `pct_accounts_above_cost_cap_single_day`)
  - `params` (day / start / end)
  - `targets` (metric + true value)
  - `metadata` (e.g. `expected_top_service`, `cap_usd`, `expected_account_id`, `price_usd`)

## Scoring Levels

Use **0–3** for explanation fidelity per QA example.

- **3 – Fully faithful**
  - All **claims match the ground truth** within the scenario’s tolerance:
    - Correct **metric** (e.g. total cost vs services‑only vs margin vs percent of accounts).
    - Correct **direction/comparison** (e.g. which service/account is “highest”, which side of cap, net gain vs loss).
    - Any **numbers** are either:
      - Explicitly correct within the tolerated error, or
      - Omitted/qualitative but still consistent (e.g. “a small share of accounts” when the true value is ≈7%).
    - Correct **driver/dimension** (e.g. `amocrm_call` is called out as top service; the right `account_id` when mentioned).
    - Correct **time window** (correct day or start–end range and no relative dates).
  - No hallucinated extra drivers, thresholds, or dependencies that contradict data.

- **2 – Mostly faithful (minor issues)**
  - Core story is **correct and grounded**, but with **small issues that do not change the main conclusion**, for example:
    - Slight numeric rounding or paraphrase (“about 11.3k” for 11,254.94; “roughly one in fourteen accounts” for 6.8%).
    - Slightly vague attribution that is still consistent (e.g. “cost cap around \$30” when `cap_usd` is 30.0).
    - Missing secondary numeric detail that is present in the target but not essential (e.g. omits the exact cap value but clearly explains that “a small share of accounts exceeded the cost limit on June 20, 2025”).
  - No clear contradictions on **metric, driver, direction, or window**.

- **1 – Partially faithful (important but incomplete / partially wrong)**
  - Explanation contains **some correct core elements**, but also **material problems**, for example:
    - Correct **driver** but wrong **direction/magnitude** (“amocrm_call had only a small contribution” when it is actually the top service).
    - Correct **day/window**, but confuses **metric** (e.g. mixing `total_cost` with `total_cost_services` or describing percent of accounts as an absolute count).
    - Mentions the correct **cap** or threshold but misstates how many / what share of accounts crossed it in a way that changes interpretation.
    - Omits a **key driver or condition** that the ground truth is explicitly about (e.g. explanation talks only about total cost but never mentions that the question was about **margin per active account**).
  - Still shows clear evidence of being grounded in the right scenario, not fully hallucinated.

- **0 – Unfaithful / incorrect**
  - Any of the following:
    - **Wrong driver** (e.g. claims `dialog` is the top cost driver when ground truth driver is `amocrm_call`).
    - **Wrong direction** (e.g. says margin “increased” when ground truth net user change is negative, or says “most accounts are over cap” for 6.7% over cap).
    - **Hallucinated numbers** not supported by the scenario (especially new big integers/precise amounts unrelated to `targets` and metadata).
    - **Wrong metric or outcome** (margin vs total cost; count vs share; using p95 as if it were a cap).
    - **Wrong time window** (different date or range; uses “yesterday/last week/this month” instead of explicit 2025 dates).
    - Explanation claims the query is **supported** while the case is a `negative_case` that should be UNSUPPORTED, or vice versa.
  - Overall narrative would **mislead a human analyst** about what actually happened in the underlying data.

## Common Error Categories

Use these tags alongside the numeric score to diagnose failures (can apply multiple):

- **hallucinated_numbers**
  - Explanation introduces **new numeric facts** not derivable from `targets`/`metadata`:
  - Examples: invented account IDs, extra thresholds, invented p-values, “thousands of accounts” when there are tens.

- **wrong_metric**
  - Confuses **which KPI** is being discussed:
  - Examples: talks about “revenue” or “profit” instead of cost; mixes `total_cost` with `total_cost_services`; explains “number of accounts over cap” when the target is a **percentage**.

- **wrong_driver**
  - Misidentifies the **main driver or dimension**:
  - Examples: top service is actually `amocrm_call` but explanation focuses on `dialog` as the biggest contributor; wrong `account_id` called out as top account.

- **wrong_direction**
  - Gets the **sign or comparison** wrong:
  - Examples: says “margin is low” when it is actually high; “most accounts are above cap” when only a small share are; “user base grew” when net user change is negative.

- **wrong_window**
  - Uses incorrect or relative **time references**:
  - Examples: “yesterday”, “last week”, month‑to‑date, or dates outside 2025 when the scenario is explicitly anchored on a given 2025‑MM‑DD or window.

- **missing_key_driver**
  - Omits a **central element** needed for a faithful explanation:
  - Examples: for `top_service_by_cost_single_day`, explanation never mentions the actual top service; for `pct_accounts_above_cost_cap_single_day`, explanation never mentions the cost cap concept.

- **unsupported_but_treated_as_supported**
  - For `negative_case` records: explanation answers the question as if it were supported instead of clearly signalling an UNSUPPORTED request (e.g. relative dates, out‑of‑year dates, unsupported metrics/dimensions).

## Short Examples

- **Example 1 – Top service by cost (score 3)**
  - Ground truth: `type="top_service_by_cost_single_day"`, `params.day="2025-06-20"`, `metadata.expected_top_service="amocrm_call"`.
  - Explanation: “On 20 June 2025, **amocrm_call** was the largest cost driver, accounting for the highest spend among all services that day.”
  - Assessment: **3 (fully faithful)**, no error tags.

- **Example 2 – Percent of accounts above cap (score 2)**
  - Ground truth: `type="pct_accounts_above_cost_cap_single_day"`, value ≈0.0676, `metadata.cap_usd=30.0`.
  - Explanation: “On 2025‑06‑20, only a **small share of accounts (around 7%)** exceeded the **\$30 cost cap**; most accounts stayed within the limit.”
  - Assessment: **3** if the approximate 7% is acceptable; **2** if you treat the qualitative “around 7%” as slightly underspecified but directionally fine. No error tags.

- **Example 3 – Wrong driver and direction (score 0, tags)**
  - Ground truth: same as Example 1 (top service is `amocrm_call`).
  - Explanation: “On June 20, 2025, **dialog** barely contributed to total costs, while other services were insignificant.”
  - Assessment: **0 (unfaithful)** with tags: `wrong_driver`, `wrong_direction` (dialog is wrongly elevated and described as dominating or negligible in a way that contradicts data).

- **Example 4 – Negative case treated as supported (score 0, tags)**
  - Ground truth: `negative_case`, `category="relative_date"`, expected `planner_status="UNSUPPORTED"`.
  - Explanation: “Yesterday we spent **\$11,000 in total**, which is slightly above our usual level.”
  - Assessment: **0**, tags: `unsupported_but_treated_as_supported`, `hallucinated_numbers`, `wrong_window`.

## Optional Claim Schema

For more structured auditing or automated consistency checks, each atomic explanation statement can be mapped to a **claim schema** aligned with the fact records:

- **metric**
  - What is being measured. Examples: `sum_total_cost`, `total_cost_services`, `pct_accounts_above_cap`, `avg_margin_per_account_day`, `p95_account_day_total_cost`, `new_users`, `churned_users`, `net_user_change`.

- **subject / driver**
  - The entity or dimension the metric is about:
  - Examples: service (`dialog`, `tasks`, `classifications`, `amocrm_call`), `account_id`, “all active accounts”, “accounts above cap”.

- **window**
  - Time coverage of the claim:
  - Either a single day (`day`) or a start–end window (`start`, `end`) in 2025, consistent with `params`.

- **direction / comparison**
  - Whether the claim describes being **higher/lower/top/bottom**, a threshold crossing, or a net change:
  - Examples: “top service”, “largest account”, “above cap”, “small share”, “net loss of users”.

- **magnitude**
  - Optional numeric content (value or proportion) or coarse bucket:
  - Examples: explicit values (`11254.94`, `0.0676`), or qualitative levels (“small share”, “most accounts”, “large spike”).

- **threshold / reference (optional)**
  - Any explicit caps or reference points used:
  - Examples: `cap_usd=30.0`, comparison to zero net change, previous day, or typical/average levels.

An explanation is **high‑fidelity** when all of its claims, mapped via this schema, are **consistent with the corresponding fact/negative‑case record** and do not introduce contradictory or unsupported information.

