# An AI Agent for Cost Monitoring and Explanation in Subscription-Based Platforms: System Design and Evaluation 

## Overview
This repository contains the artefact developed for the dissertation “An AI Agent for Cost Monitoring and Explanation in Subscription-Based Platforms: System Design and Evaluation.” The system combines deterministic cost analytics with LLM-supported planning and explanation to monitor daily usage-cost behaviour, detect anomalous patterns, generate scheduled reports, and answer follow-up analytical questions. It is designed around a shared Evidence Pack contract, which allows dashboards, summaries, and generated answers to remain grounded in explicit numerical evidence.

## Research context / purpose
Public datasets are poorly suited to this problem setting because they rarely reflect AI subscription-platform cost structures and typically do not provide labelled anomaly scenarios needed for controlled evaluation. This project therefore uses a synthetic, deterministic usage-cost dataset with injected anomalies and ground-truth labels. The repository supports both the implementation of the artefact and the reproducible evaluation of its behaviour across three main dimensions: anomaly detection performance, explanation fidelity, and end-to-end pipeline integration. In this sense, the repository is both a working prototype and a research instrument used to generate the dissertation’s tables, figures, and run artefacts.

## Key features
- **Synthetic dataset generation**: configurable usage, cost, and anomaly-injection logic.
- **Schema-validated CSV backend**: constrained structured querying and lineage metadata.
- **Deterministic KPI computation**: standardized Evidence Pack construction.
- **Deterministic daily reporting**: dashboard PNG and textual summary generation.
- **LLM-based Analyst**: evidence-grounded narrative explanation.
- **Deterministic Verifier**: numeric and directional fidelity checks.
- **Planner-driven interactive Q&A pipeline**: safe plan compilation into backend queries.
- **Scenario-based evaluation**: anomaly detection, explanation fidelity, and integrated pipeline behaviour.
- **Run-level artefact logging**: traceability, reproducibility, and dissertation evidence.


## Repository structure

- `src/cost_agent_mvp/`
  Main application code.
  - `agent/` — orchestration, planner, analyst, verifier, detector, prompt loading
  - `analytics/` — KPI definitions and Evidence Pack construction
  - `data/` — schema validation and CSV backend
  - `reports/` — daily report runner and summary generation
  - `viz/` — dashboard rendering and chart specifications
  - `ui/` — Telegram interfaces
  - `cli/` — command-line entry points
  - `core/` — shared run metadata and utility structures

- `configs/`
  YAML configuration files, including semantic-layer constraints and generator/report settings.

- `data/`
  Synthetic datasets, scenario datasets, labels, and related metadata.

- `outputs/`
  Generated run artefacts, including dashboards, summaries, evidence CSVs, verifier outputs, and run records.

- `tests/`
  Automated tests for contracts, analytics, orchestration, and evaluation components.

- `docs/`
  Supporting project documentation such as workflow notes, development guidance, and artefact specifications.

  ## Requirements
  The project was developed in Python and is designed to run as a lightweight research prototype with optional LLM- and Telegram-based components.

  ### Core runtime requirements

  - **Python**: 3.12+
  - **Package manager**: `pip` for installation.
  - **Git**: for cloning the repository.

  The runtime environment uses the following main libraries:

  - **pandas, numpy**: deterministic data processing and KPI computation.
  - **matplotlib**: dashboard rendering to static PNG.
  - **PyYAML**: configuration loading.
  - **pydantic**: structured models and validation.
  - **python-dotenv**: environment variable loading.
  - **openai**: OpenAI-compatible LLM client for planner and analyst modes.
  - **aiogram**: Telegram bot interface.
  - **openpyxl**: spreadsheet export/import utilities where needed.
  - **tqdm**: progress reporting in longer runs.

  For local development and reproducibility checks:

  - **pytest, pytest-cov, pytest-mock**: automated testing.
  - **ruff**: linting and formatting.
  - **pre-commit**: local quality guardrails.
  - **coverage, build**: test coverage and packaging support.

  ### External services
  Some workflows require additional credentials:

  - **LLM-backed modes**: require an API key and, if applicable, a custom base URL for an OpenAI-compatible provider.
  - **Telegram modes**: require a Telegram bot token.

  Deterministic reporting and most evaluation scripts can still be run without Telegram, and some parts can be run without live LLM access if mock modes are enabled.


## Installation

1. Clone the repository
```bash
git clone https://github.com/shakhovak/Agent-augmented-Cost-Analytics-MSc-.git
cd agent_augmented_cost_analytics
```

2. Create and activate a virtual environment
Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install runtime dependencies
```bash
pip install -r requirements.txt
```
4. Install development dependencies (optional)
```bash
pip install -r requirements-dev.txt
```
5. Verify the setup
A simple way to verify the environment is to run tests and linting:
```bash
pytest
ruff check .
```
If your project includes a diagnostic command, you can also run:
```bash
python -m src.cost_agent_mvp.cli.doctor
```
6. Optional: enable pre-commit hooks
```bash
pre-commit install
```
This helps catch formatting and linting issues before commits.


## Configuration

Project behaviour is controlled through a combination of environment variables, YAML configuration files, and prompt templates. Installation prepares the environment; configuration determines how the system behaves at runtime.

### Configure environment variables

Create a `.env` file in the project root and add the variables needed for the modes you intend to use.

Example:
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_provider_base_url_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```
Only include the variables required for your setup. Keep `.env` out of version control.

### YAML configuration files

Most non-secret project settings are defined in YAML files under `configs/`. These configuration files control the behaviour of the main system components.

Typical configuration areas include:

- **Synthetic data generation**: population size, date range, account activity, service mix, pricing rules, and anomaly-injection parameters.
- **Semantic layer / query constraints**: allowed metrics, dimensions, filters, and execution limits for planner-driven Q&A.
- **Daily reporting templates**: reporting windows, top-N settings, and other parameters used in scheduled reporting.

Where relevant, YAML configuration is used to support reproducibility by separating experimental settings from code.

### Prompt templates

LLM prompts are stored separately from implementation logic and loaded through the prompt loader. This allows prompt wording to be versioned and updated without changing core orchestration code.

Prompt templates are used by:

- **Planner**: converts free-text questions into structured analytical plans.
- **Analyst**: generates evidence-grounded narrative explanations.
- **Verifier prompt layer**: where applicable, provides structured explanation-checking support.

This separation helps make LLM behaviour more transparent and easier to compare across iterations.

## Dataset creation

The synthetic dataset is created in two steps: **base usage–cost generation** and **anomaly injection**.

- **1. Generate the base synthetic dataset**
  - Configuration: `data/config_data_generator.yaml` (controls horizon, accounts, lifecycle, and cost model; output path is set under `output.path`).
  - Run:
    ```bash
    python data/generator.py --config data/config_data_generator.yaml
    ```
  - This will write a schema-validated joint usage–cost CSV (by default `data/joint_info_2025.csv` as specified in the config).

- **2. Inject labelled anomaly scenarios**
  - Configuration: `data/anomaly_scenarios.yaml` (defines anomaly types, windows, and sampling logic).
  - Run, pointing to the base CSV from step 1:
    ```bash
    python data/inject_anomalies.py \
      --input data/joint_info_2025.csv \
      --config data/anomaly_scenarios.yaml \
      --output data/joint_info_2025_with_anomalies.csv \
      --labels data/anomaly_labels.jsonl
    ```
  - The output CSV is used by the evaluation scripts, and the JSONL file contains ground-truth anomaly labels and metadata for each injected scenario.

## Main workflows

### Daily report

Run a daily reporting job that reads the synthetic CSV backend, computes KPIs, and writes dashboards plus textual summaries into `outputs/`. You can use:

- **Telegram bot mode (operator-facing)**
  - Script: `src/cost_agent_mvp/ui/tg_bot_daily_llm_report_QA.py`.
  - Configure the following environment variables (for example in `.env`):
    - `TELEGRAM_BOT_TOKEN`, `CSV_PATH`, `OUTPUTS_ROOT`, `DATASET_NAME`
    - `REPORT_TEMPLATES_PATH`, `REPORT_TEMPLATE_ID`
    - `SEMANTIC_LAYER_PATH`, `PROMPTS_DIR`
    - Optional narrative settings: `REPORT_MODE=deterministic|llm`, `MOCK_LLM=0|1`
  - Run the bot:
    ```bash
    python src/cost_agent_mvp/ui/tg_bot_daily_llm_report_QA.py
    ```
  - In Telegram, send `/start`, then use:
    - **📊 Daily KPI report (latest)** to run the report for the latest available date.
    - **📅 Daily KPI report (pick a date)** to select a specific date from the dataset.
  - The bot will generate a dashboard PNG and summary text under `OUTPUTS_ROOT` and send them back to the chat. `REPORT_MODE` controls whether the summary is purely deterministic or includes an LLM-generated narrative.

### Interactive Q&A

Use the planner–analyst pipeline to answer ad hoc analytical questions against the same semantic layer and Evidence Pack contracts. There are two main ways to use it:

- **Telegram bot mode (interactive analyst)**
  - Reuse the same bot from the daily-report workflow (`tg_bot_daily_llm_report_QA.py`).
  - In Telegram, click **🧠 Chat with companion** or simply type a free-text question such as:
    - “What was total cost on 2025-06-20?”
    - “Top 5 accounts by cost for 2025-07-20.”
  - The bot:
    - Uses the latest available dataset date as `anchor_day`.
    - Calls `Orchestrator.run_ad_hoc` with `mode="llm"` (or deterministic if `MOCK_LLM=1`).
    - Returns the analyst answer, annotated with verifier status (e.g. unsupported or low-confidence).

- **CLI diagnostic mode (pipeline debugging)**
  - Script: `evalaution/diagnose_llm_pipeline.py`.
  - Ensure these environment variables are set: `CSV_PATH`, `SEMANTIC_LAYER_PATH`, `PROMPTS_DIR`, `YANDEX_CLOUD_API_KEY`, `YANDEX_CLOUD_FOLDER` (and any Yandex/OpenAI base URL model settings you use elsewhere).
  - Run:
    ```bash
    python evalaution/diagnose_llm_pipeline.py
    ```
  - The script:
    - Logs the full planner/analyst calls to the LLM (inputs and truncated outputs).
    - Shows the signatures of the plan compiler and `Orchestrator.run_ad_hoc` to verify keyword arguments such as `anchor_day`.
    - Executes a single representative question end-to-end, so you can confirm that the semantic layer, prompts, and orchestrator wiring are functioning correctly without going through Telegram.

### Evaluation

The evaluation pipeline is built around **scenario suites** that combine factual questions, unsupported (negative) cases, and LLM-generated phrasings for each injected anomaly event.

- **1. Build a scenario suite for injected anomalies**
  - Input: injected CSV and labels from the dataset creation step (for example `data/joint_info_2025_with_anomalies.csv` and `data/anomaly_labels.jsonl`).
  - Script: `evalaution/build_scenario_suite.py` (orchestrator).
  - Requirements:
    - LLM credentials for question generation (Yandex Cloud OpenAI-compatible API), as used by the analyst.
    - A question-generation prompt file, e.g. `prompt_for_questions.txt`.
  - Typical command:
    ```bash
    python evalaution/build_scenario_suite.py \
      --csv data/joint_info_2025_with_anomalies.csv \
      --labels data/anomaly_labels.jsonl \
      --out-root outputs/scenario_suites \
      --suite-id mvp_v0 \
      --prompt prompt_for_questions.txt \
      --n-variants 3 \
      --temperature 0.2
    ```
  - This will:
    - Generate per-scenario **positive facts** aligned to each anomaly window.
    - Call `evalaution/generate_negative_cases.py` to add **negative UNSUPPORTED cases**.
    - Call `evalaution/generate_question_variants.py` to add LLM-backed **question variants** for each positive fact.
    - Write per-event folders under `outputs/scenario_suites/<suite-id>/` with:
      - `scenario_config.json` and `labels.json` (metadata and anomaly labels),
      - `qa/positives.jsonl`, `qa/negatives.jsonl`, and `qa/suite.jsonl` (combined question set for evaluation).

- **2. (Optional) Run the fact sampler directly**
  - If you want to generate a generic fact set not tied to anomaly windows, you can call `evalaution/generate_fact_scenarios.py` directly:
    ```bash
    python evalaution/generate_fact_scenarios.py \
      --csv data/joint_info_2025_with_anomalies.csv \
      --out outputs/facts/facts.jsonl \
      --n 50 \
      --seed 42
    ```
  - You can then:
    - Use `evalaution/generate_question_variants.py` on `facts.jsonl` to attach question variants.
    - Use `evalaution/generate_negative_cases.py` to create a shared pool of unsupported queries.

Downstream evaluation scripts (for anomaly detection and explanation fidelity) consume the scenario suite outputs (especially the `qa/suite.jsonl` files and manifest under `outputs/scenario_suites/`) to score planner and analyst behaviour in a fully reproducible way.

- **3. Run QA fidelity evaluation on a single suite**
  - Script: `evalaution/eval_scenarios_fidelity.py` (question-level fidelity and planner metric evaluation).
  - Typical command (per `qa/suite.jsonl` produced in step 1):
    ```bash
    python evalaution/eval_scenarios_fidelity.py \
      --facts outputs/scenario_suites/mvp_v0/<event_id>/qa/suite.jsonl \
      --csv-path data/joint_info_2025_with_anomalies.csv \
      --semantic-layer-path configs/semantic_layer.yaml \
      --prompts-dir configs/prompts \
      --outputs-root outputs/eval_questions \
      --suite-id mvp_v0
    ```
  - This records per-question runs (planner + analyst traces, numeric extraction, planner metrics) and writes a `summary.csv` and `summary.jsonl` under `outputs/eval_questions/mvp_v0/`.

- **4. Evaluate detector + QA jointly across the whole scenario bundle**
  - Script: `evalaution/eval_scenario_bundle.py` (RQ1–RQ3 bundle-level analysis).
  - Typical command:
    ```bash
    python evalaution/eval_scenario_bundle.py \
      --suite-root outputs/scenario_suites/mvp_v0 \
      --semantic-layer-path configs/semantic_layer.yaml \
      --prompts-dir configs/prompts \
      --mode llm
    ```
  - This:
    - Reuses the deterministic detector to score event-level anomaly detection (hits, delay, nearby false positives).
    - Runs (or reuses) QA evaluation per scenario via `evalaution/eval_scenarios_fidelity.py`.
    - Computes simple integration metrics (how often detector context is injected and whether flagged days are mentioned).
    - Writes per-scenario `scenario_metrics.json` and a suite-level `bundle_results.csv` / `bundle_results.jsonl`.

- **5. Evaluate account-level anomaly attribution across label events**
  - Script: `evalaution/eval_anomalies_accounts.py` (account precision/recall against labels).
  - Typical command:
    ```bash
    python evalaution/eval_anomalies_accounts.py \
      --labels data/anomaly_labels.jsonl \
      --csv data/joint_info_2025_with_anomalies.csv \
      --out-root outputs/anomaly_eval \
      --detector-script src/cost_agent_mvp/cli/run_detector_7d.py
    ```
  - This runs the detector for each labelled event/day, compares flagged accounts to the ground-truth fields in `anomaly_labels.jsonl`, and writes per-event and per-day summaries (`results_event.csv` and `results_day.csv`), optionally mirroring per-event metrics back into the corresponding scenario suite folders.


## License

This repository is provided for academic and research purposes in support of the MSc dissertation “An AI Agent for Cost Monitoring and Explanation in Subscription-Based Platforms: System Design and Evaluation.”

Unless explicitly stated otherwise in a separate `LICENSE` file or per-file notice, all rights are reserved by the author. You may:

- **Read and run the code locally** to reproduce the experiments, figures, and tables in the dissertation.
- **Adapt the ideas and configuration patterns** for your own internal research work, with appropriate citation.

You may not:

- **Redistribute the repository or derivative works** as a commercial product or SaaS offering.
- **Use the artefact or its prompts** to build a competing commercial service without prior written permission from the author.

If you intend to use this artefact beyond personal or academic research (for example, in teaching materials, open-source derivatives, or commercial prototypes), please contact the author to discuss licensing and attribution.