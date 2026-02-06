# Ethics Statement

## Summary
This MSc research project uses **synthetic data only** generated specifically for benchmarking and evaluation. The work does not involve human participants, personal data, or real customer communications.

## Data sources
- All datasets used in experiments are generated programmatically.
- The dataset does not contain names, email addresses, phone numbers, message content from real users, or other personal identifiers.

## Human participants
- No human participants are recruited.
- No user studies are conducted.
- No interventions are performed on real individuals or real systems.

## Risks and mitigations
Potential risks are primarily related to:
- misuse of the tooling for non-synthetic data, and
- accidental exposure of credentials.

Mitigations:
- clear repository guidance that datasets must remain synthetic,
- strict secrets handling (no keys committed; environment variables / `.env`),
- restricted access controls for any remote deployments.

## Intended use
The system is intended for academic evaluation of anomaly detection and explanation workflows. Any adaptation to real customer data would require:
- a separate ethics review,
- a privacy impact assessment,
- appropriate consent/legal basis, and
- additional technical controls (access control, anonymization, auditing).
