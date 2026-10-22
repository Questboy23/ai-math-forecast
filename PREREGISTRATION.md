# PREREGISTRATION: AI-resolution forecasting of open Erdős problems

Written before any test-period model metric was computed. The file's SHA-256 and Eastern timestamp are in state/prereg_hash.txt and at the top of REPORT.pdf.
What was seen before this file: label counts per origin (events, competing events), the Haiku contamination probe, feature distributions. No AUC, PR-AUC, Brier, calibration, or any model fit on test-period rows.

## Question
Do obscurity and attention features predict which open Erdős problems get resolved with AI involvement within 3 months (H1), and does Euled "bridge distance" add predictive power beyond them (H2)?

## Unit, origins, risk set
- Row = (problem, origin). Origins: the 1st of each month, 2025-09-01 through 2026-07-01 (11). Spec origins 2025-07-01 and 2025-08-01 are dropped because problems.yaml history starts 2025-08-31 (see DECISIONS.md).
- Test origins: 2026-01-01, 02-01, 03-01, 04-01, 05-01, 06-01, 07-01 (7). Train-only origins: 2025-09-01 to 2025-12-01.
- Risk set: informal_status.state == "open" at the origin (git snapshot of teorth/erdosproblems data/problems.yaml, last commit before the origin).
- Event y=1: state is resolved (not open, not falsifiable/verifiable/decidable) at origin+3 months AND the AI-contributions wiki has a green or yellow row in sections 1a-1d or 2a dated within [resolution date - 120 days, resolution date + 30 days]. Resolution date = status last_update, clipped to the window.
- Competing event: resolved in the window without such a row. Row counts as y=0 at its origin; the problem leaves later risk sets because it is no longer open.
- Event types: `new_sol` (sections 1a-1d) and `lit_id` (section 2a only).

## Features (all as of the origin)
Obscurity and attention (OBS): log prize; age of earliest Erdős-key reference (origin year minus year); Lean file in google-deepmind/formal-conjectures before the origin and its age in days; site "formalized" flag; log statement length; number of status edits across monthly snapshots up to the origin; days since last status update; OEIS link count; tag count and 0/1 tag indicators (tags with variance); number of AI-wiki rows (any outcome, any section) for the problem dated before the origin.
Bridge: d_near, d_mass, d_2hop, concept_cost (definitions in SPEC.md), plus hub (Hub score). Corpus = arXiv math primary categories plus cs.DM/IT/CC/CG/LO/DS, math-ph, stat.TH, submitted before the origin (versions[0].created), embedded with all-MiniLM-L6-v2 (title + first 600 characters of abstract). Leakage filter: documents mentioning Erdős with a problem number, and arXiv ids cited on problem pages or the AI wiki, are removed for every problem and origin.
Dropped: OpenAlex citation counts (source papers cannot be resolved reliably), comment counts (no dated snapshots), reference-list size (current lists include post-resolution additions).
Side test only, never in a headline model: Haiku technique-standardness rating 1 to 7. Contamination probe result in DECISIONS.md.

## Models (train only on rows whose origin+3 months is on or before the test origin; expanding window)
1. random: uniform scores, seed 0.
2. lr_obs: standardized logistic regression, L2, OBS features. C from {0.003..3} by inner time-split CV (log-loss; default 0.1 if fewer than 4 training origins).
3. lr_obs_bridge: same with OBS + bridge + hub.
4. hgb_all: HistGradientBoosting (depth 3, lr 0.05, 150 iterations, L2 1.0, min leaf 20), all features, no tuning.

## Hypotheses and decision rules
- H1 supported iff pooled out-of-time AUC of lr_obs > 0.60 (point estimate); the 95% CI is reported with it.
- H2 supported iff AUC(lr_obs_bridge) > AUC(lr_obs) and the two-sided DeLong p < 0.05 on pooled test predictions. Also reported: permutation test (bridge columns shuffled within origin across all rows, 300 shuffles, C fixed at the real run's values) and a problem-clustered bootstrap CI of the AUC difference. Rows are correlated (a problem appears in up to 3 consecutive origins), so DeLong p is optimistic and the clustered CI and permutation p are the safer readings [medium].
- A null H2 is reported as a finding.

## Metrics (pooled test predictions)
AUC with 2,000-sample bootstrap CI resampling problems; PR-AUC next to the base rate; Brier; 10-bin calibration table; precision@20 averaged over test origins.
Secondary: rerun with `lit_id` events only and `new_sol` events only (rows with the other type's event are dropped). Robustness: drop each feature group in turn from lr_obs_bridge and hgb_all, report AUC change.

## Forecast
Refit on all rows. Forecast model = whichever of lr_obs, lr_obs_bridge, hgb_all has the lowest pooled test Brier score. P(AI-resolved within 3 months) = model output at origin 2026-10-22 for problems open then. P(12 months) = 1 - (1 - P3)^4, which assumes constant hazard [low]. Scoring rule for later checks: Brier score on 2027-01-22, 2027-04-22, 2027-10-22.

## Not allowed after this point
Changing origins, features, models, labels, or hypotheses after seeing test metrics.
