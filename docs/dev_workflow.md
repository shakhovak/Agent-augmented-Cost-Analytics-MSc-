# Development workflow

This repository follows a lightweight Agile/iterative prototyping workflow:
short build–test–refine cycles, each producing a working increment and evidence
(tests + CI logs + run artifacts).

## 1. Branching strategy (one PR per issue)

- `main` is always kept in a working state.
- For each Issue, create a short-lived feature branch:
  - Naming: `m0-xx-short-name` (milestones) or `m1-xx-short-name` (next phases).
- Work is merged via Pull Request (PR). No direct commits to `main`.

Typical commands:

```bash
git checkout main
git pull
git checkout -b m0-07-doctor


---
