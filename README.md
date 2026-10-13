# Machine Learning for NBA Injury Risk Prediction

This repository is being developed from an NBA course project in separately approved stages. **Stage 4 is complete: four fixed baselines compared with three chronological CV folds and the frozen validation set.** No tuning or held-out test evaluation has been performed. Read [the Stage 4 report](docs/stage4_report.md), [Stage 3 report](docs/stage3_report.md), and [feature dictionary](docs/feature_dictionary.md).

Random Forest has the highest validation Average Precision (**0.2879**, versus prevalence **0.1367**), with ROC-AUC **0.6888**. Logistic Regression has the highest validation ROC-AUC (**0.6937**) and slightly higher mean temporal CV AP than Random Forest. The small validation AP difference does not establish a robust winner. These are validation results for the restricted recorded-event benchmark, not final test or clinical prediction results. See [the comparison table](results/tables/stage4_model_comparison.csv).

The proposed benchmark predicts a documented injury-related inactive-list episode within 14 days of a player-date snapshot, using a delayed historical cutoff. It has **9,804 labelable observations across 691 players**, with **13.58%** positive outcomes. These are cohort counts, not model performance. The data cannot establish a clean, continuously observed NBA at-risk roster or unbiased clinical injury probabilities. See [the Stage 2 design and limitations](docs/prediction_design.md) before continuing.

The original analysis examined injury-log trends, positions, teams, pre-injury performance, and a season-level logistic regression. Its saved ROC-AUC of **0.5363** uses a random split and full-season information. It is a historical course result and does not establish prospective injury prediction performance.

Read [the source review](docs/original_project_review.md) for the actual data, methods, limitations, and work needed. [The migration manifest](docs/migration_manifest.json) records all copied files and SHA-256 checksums. [The repository plan](docs/repository_plan.md) distinguishes the current layout from future work.

## Current contents

- `data/raw/`: unchanged primary CSVs and one supplementary archive, retained locally and ignored by Git.
- `notebooks/legacy/` and `src/legacy/`: unchanged course notebooks and script. They still contain historical absolute paths and are references, not portable entry points.
- `docs/legacy/`: original report and slides, retained locally and ignored by Git.
- `results/legacy/`: saved notebook figures and outputs, extracted without rerunning the analysis.
- `src/nba_injury/data/` and `scripts/build_stage2_design.py`: audit log semantics and identities, generate spaced observations, and assign chronological splits.
- `docs/prediction_design.md`, `docs/stage2_design_config.json`, and `results/tables/stage2_*`: the Stage 2 design, frozen rules, and audits.
- `data/processed/stage2/`: generated observations, labels, episodes, and split files, retained locally and ignored by Git.
- `scripts/verify_migration.py`: verifies copied source bytes without training or modifying data.
- `src/nba_injury/features/` and `scripts/build_stage3_features.py`: workload/performance/episode-history features and unfitted sklearn preprocessing factories.
- `results/figures/stage3/` and `results/tables/stage3_*`: six EDA figures, dictionaries, missingness/redundancy audits, and summaries.
- `data/processed/stage3/`: X/y matrices and separate row manifests/date provenance, local and ignored by Git.
- `scripts/run_stage4_baselines.py`, `requirements-stage4.txt`, and `docs/stage4_report.md`: fixed model settings, three temporal folds with at least 23-day separation and label-availability purging, and results.
- `results/tables/stage4_model_comparison.csv` and `results/figures/stage4/`: one comparison CSV and two validation curve figures.
- `data/processed/stage4/validation_predictions.csv`: local validation scores and fixed-threshold predictions, ignored by Git.

## Reproduce completed stages

From the repository root, with Python 3:

```bash
python3 scripts/verify_migration.py
python3 scripts/build_stage2_design.py
python3 scripts/build_stage3_features.py
python3 -m unittest discover -s tests -v
git status --short
```

The verifier uses the Python standard library. Select a Python environment with the dependencies in `requirements.txt`; shell `python3` can resolve to different installations. Stage 3 ran with the existing Anaconda Python 3.13.5 and scikit-learn 1.6.1. Stage 4 uses an ignored local environment with precompiled XGBoost 3.4.1 and LightGBM 4.7.0; tested scientific package versions are pinned separately in `requirements-stage4.txt`. Stage 4 fits preprocessing separately on each fold's training rows.

To reproduce Stage 4 from the already generated frozen Stage 3 files:

```bash
/opt/anaconda3/bin/python3 -m venv --system-site-packages .venv-stage4
.venv-stage4/bin/python -m pip install --only-binary=:all: --no-deps xgboost==3.4.1 lightgbm==4.7.0
.venv-stage4/bin/python scripts/run_stage4_baselines.py
```

## Next stage

Stage 5 requires explicit approval before tuning or final test evaluation. Four-model baseline comparison and chronological CV are now supported by reproducible code and validation results; tuning and final test performance remain future work. The restricted population and ascertainment limitations in the Stage 2 design continue to apply.
