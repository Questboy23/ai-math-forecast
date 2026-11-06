# AI Math Forecast: a dated public forecast of which open Erdős problems AI will resolve, with a preregistered backtest.

**593 open problems forecast. The backtest found no skill (AUC 0.57, 95% CI 0.51 to 0.63).**

Dated public forecast of which open Erdős problems will be resolved with AI involvement, plus the preregistered backtest behind it.

**Backtest result (7 test origins, 2026-01 to 2026-07): H1 not supported (AUC 0.571, CI 0.511 to 0.632). H2 not supported (bridge distance: AUC change -0.004, DeLong p 0.54, permutation p 0.61).** The forecast model (gradient boosting) had pooled test AUC 0.490, so its ranking has no demonstrated skill [low]. Treat the probabilities as base-rate estimates, not a ranking.

## Files
- forecast.csv: 593 problems open on 2026-10-22, sorted by P(AI-resolved within 3 months). 12-month column assumes constant hazard [low].
- PREREGISTRATION.md (SHA-256 in prereg_hash.txt), REPORT.pdf, EPOCH_SUMMARY.md, DECISIONS.md, PREFLIGHT.md, AUDIT_LABELS.md, code/.

## Scoring rule
Brier score of p_ai_resolved_3mo against the outcome (open problem leaves "open" with a green/yellow AI wiki entry per PREREGISTRATION.md) over the 3 months after 2026-10-22. Checked 2027-01-22, 2027-04-22, 2027-10-22 (the 12-month column is scored on the last date).

## forecast.csv SHA-256
2c9a9d86ec22b205717723ce04ad9d8d6e013372f0f79b897e6813b3f1245981
