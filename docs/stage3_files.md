# Stage 3 files

Modified existing file: `README.md`. All original course material, migrated raw files, and frozen Stage 2 code/configuration/cohorts remain unchanged. Requirements and ignore rules are unchanged.

Created:

- `src/nba_injury/features/__init__.py`
- `src/nba_injury/features/historical.py`
- `src/nba_injury/features/preprocessing.py`
- `scripts/build_stage3_features.py`
- `tests/test_stage3_features.py`
- `docs/feature_dictionary.md`
- `docs/stage3_report.md`
- `docs/stage3_verification.json`
- `docs/stage3_files.md`
- `results/tables/stage3_example_rows.csv`
- `results/tables/stage3_feature_dictionary.csv`
- `results/tables/stage3_feature_summary.json`
- `results/tables/stage3_high_correlations.csv`
- `results/tables/stage3_missingness.csv`
- `results/tables/stage3_missingness_by_phase.csv`
- `results/tables/stage3_prevalence.csv`
- `results/tables/stage3_source_missingness.csv`
- `results/tables/stage3_train_by_outcome.csv`
- `results/tables/stage3_train_distributions.csv`
- `results/tables/stage3_variance_audit.csv`
- `results/figures/stage3/01_prevalence.png`
- `results/figures/stage3/02_train_workload.png`
- `results/figures/stage3/03_train_injury_history.png`
- `results/figures/stage3/04_train_performance.png`
- `results/figures/stage3/05_missingness.png`
- `results/figures/stage3/06_train_correlations.png`
- `data/processed/stage3/X_test.csv` — local, ignored
- `data/processed/stage3/X_train.csv` — local, ignored
- `data/processed/stage3/X_validation.csv` — local, ignored
- `data/processed/stage3/excluded_candidate_audit.csv` — local, ignored
- `data/processed/stage3/row_manifest_test.csv` — local, ignored
- `data/processed/stage3/row_manifest_train.csv` — local, ignored
- `data/processed/stage3/row_manifest_validation.csv` — local, ignored
- `data/processed/stage3/source_date_audit.csv` — local, ignored
- `data/processed/stage3/y_test.csv` — local, ignored
- `data/processed/stage3/y_train.csv` — local, ignored
- `data/processed/stage3/y_validation.csv` — local, ignored

No model artifacts, fitted real-data transformations, commits, or remote changes were created. Source dictionary/summary tables and the six figures can be reviewed and committed; row-level matrices/manifests/provenance stay local.
