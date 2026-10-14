# Superseded: corrected comparison at 3e0eea0 (frozen scorers at 8a59105)

Replaced by `artifacts/pilot12b/probe_validation/` (code 3857a72). The newer code fits tuning
preprocessing inside each fold, makes the learned combination's internal reference predictions
pure, and validates frozen application by schema. Its effect on these results: ranking metrics
and decisions are identical for every scorer except R2's learned combination (round-0 panel
median 0.660 → 0.640; one partition's best-of-six 67 → 66). The frozen scorers recorded here
(SHA-256 1a9b2ed5…) are not used; the validation tranche uses 8e5785fe… (3857a72).
