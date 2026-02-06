# Security and Data Handling

This repository supports an MSc research project. The datasets used for experiments are **synthetic** (generated data) and do not contain personal data or real customer records.

## Scope

This document describes:
- local storage and access controls
- deployment/storage approach for the prototype
- secrets handling
- backup practices
- retention and deletion policy

## Local storage and access controls

- Development and experiments are run locally on the researcher’s machine.
- Project files and datasets are stored locally in a dedicated project directory.
- Local storage is protected using **full-disk encryption** (OS-level encryption) and an OS user account protected by a strong password.
- The machine is kept up to date with security patches and uses a firewall enabled by default.

## Deployment and remote execution (if applicable)

If the prototype is deployed to a remote server/VPS for demonstration or evaluation:

- Access is restricted to **SSH-only**.
- SSH uses key-based authentication; password login is disabled where possible.
- A host firewall is enabled and configured to allow only required ports (typically SSH).
- Administrative access is limited to the researcher only (principle of least privilege).
- System updates are applied regularly.

## Secrets handling (API keys, tokens)

- Secrets (API keys, tokens, credentials) **must not** be committed to git.
- Secrets are provided via environment variables or a local `.env` file.
- `.env` is excluded from version control (via `.gitignore`).


## Dataset storage policy

- Generated datasets are stored only locally (project directory).
- **No public cloud storage is used for datasets** for this project (e.g., no public buckets/drives for experiment datasets).

## Backups

- Backups (if used) are stored in an **encrypted** form (encrypted disk / encrypted archive).
- Backups are kept in a private location accessible only to the researcher.
- Backup frequency is ad hoc (e.g., before major milestones) to reduce risk of data loss.

## Retention and deletion policy

### Retention period
- Synthetic datasets, experiment outputs, and logs are retained **until grading is confirmed + 12 months** to support:
  - reproducibility checks,
  - potential minor corrections,
  - dissertation audit requests.

### Deletion after retention
After the retention window ends:
- Local datasets and experiment outputs are securely deleted.
- If backups exist, the corresponding encrypted backups are deleted as well.

Secure deletion approach:
- Delete files and remove any remaining copies from backup locations.
- Use OS-supported secure deletion mechanisms where available and appropriate.

## Compliance notes

- This project uses **synthetic data only**.
- No personal data is collected, processed, or stored.
- No human subjects are recruited or studied as part of this data handling workflow.
