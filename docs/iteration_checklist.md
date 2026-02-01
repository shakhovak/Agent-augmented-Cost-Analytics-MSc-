# Iteration checklist (Agile evidence)

Use this checklist for each issue/PR to keep the build–test–refine cycle consistent.

## Before opening a PR
- [ ] Branch created for the issue (one PR per issue)
- [ ] Local environment is active (`.venv`) and dependencies installed

## Build (working increment)
- [ ] A runnable increment exists (command/script/entrypoint)
- [ ] Behavior is demonstrable (even minimal)

## Test
- [ ] Tests added/updated (unit and/or smoke test)
- [ ] `pytest` passes locally

## Refine (style + quality)
- [ ] Ruff passes locally:
  - [ ] `python -m ruff check .`
  - [ ] `python -m ruff format --check .`

## Evidence / logging
- [ ] Run artifacts generated (if applicable):
  - [ ] `outputs/runs/<run_id>/run_record.json` created
- [ ] If not applicable, note why in PR description (e.g., “infra-only change”)

## PR hygiene
- [ ] PR description includes `Fixes #<issue>`
- [ ] CI is green (lint + format + tests)
- [ ] Issue closes automatically when PR is merged
