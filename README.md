# Machine Learning for NBA Injury Risk Prediction

This repository is being developed from an NBA course project in separately approved stages. **Stage 2 is complete: event/identity audits and a restricted chronological prediction design.** Feature engineering and new model results have not yet been developed.

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

## Reproduce Stage 2

From the repository root, with Python 3:

```bash
python3 scripts/verify_migration.py
python3 scripts/build_stage2_design.py
python3 -m unittest discover -s tests -v
git status --short
```

The verifier uses the Python standard library; Stage 2 also requires pandas and NumPy. Its final code/tests ran with existing Python 3.13.7, pandas 2.3.2, and NumPy 2.3.3. `requirements.txt` lists the original scientific imports plus JupyterLab and remains unpinned. No packages or project environment have been installed yet.

## Next approved stage

Stage 3 requires explicit approval of the task to continue with, including the restricted population and outcome-ascertainment limitations in the Stage 2 report. It would then build strictly historical features. Modeling waits for the later approved stages. Résumé claims about model comparisons, tuning, rolling features, and injury-history predictors remain goals until supported by implemented code and verified results.
