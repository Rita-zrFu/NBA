# Machine Learning for NBA Injury Risk Prediction

This repository is being developed from an NBA course project in separately approved stages. **Stage 3 is complete: 36 strictly historical predictors, leakage tests, and descriptive EDA.** No new models have been trained. Read [the Stage 3 report](docs/stage3_report.md) and [feature dictionary](docs/feature_dictionary.md).

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

## Reproduce completed stages

From the repository root, with Python 3:

```bash
python3 scripts/verify_migration.py
python3 scripts/build_stage2_design.py
python3 scripts/build_stage3_features.py
python3 -m unittest discover -s tests -v
git status --short
```

The verifier uses the Python standard library. Select a Python environment with the dependencies in `requirements.txt`; shell `python3` can resolve to different installations. Stage 3 ran with the existing Anaconda Python 3.13.5 and scikit-learn 1.6.1. See the stage reports for exact tested versions. Dependencies remain unpinned, and no packages or project environments have been installed here. Stage 3's preprocessors are unfitted; future CV must fit them within each training fold.

## Next approved stage

Stage 4 requires explicit approval before baseline/model comparisons begin. Workload, recent performance, and documented episode-history feature engineering are implemented. Résumé claims about model comparisons, cross-validation, tuning, and prediction performance remain goals until supported by code and verified results. The restricted population and ascertainment limitations in the Stage 2 design continue to apply.
