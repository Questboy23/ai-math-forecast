# Label audit (Stage 1), 01:47 EDT
Sample: 25 rows drawn at random (seed 7) from the 340 rows that are non-trivial (event=1 or competing=1), because the ~7,000 plain negatives are trivially correct. Each was checked against the wiki rows for that problem (section, date, outcome emoji) and the yaml status change date.
Result: 25 of 25 consistent with the wiki text (100%) [medium]. 13 events, 12 competing. Checks passed: every event has a green/yellow row in 1a-1d or 2a inside [resolution-120d, resolution+30d]; every competing row either has no AI row or only formalization (2b), red, or out-of-window rows.
Not done: opening each of the 25 problems on erdosproblems.com to confirm the status change itself. The status field is the maintainers' data, so the audit covers the AI-attribution rule only [medium].
Known weak spot: the 120-day lookback can attach an older AI row to a later human resolution (example #347: AI row 88 days before resolution). Parser kept (agreement above 90%).
