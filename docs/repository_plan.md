# Portfolio repository plan

Stage 1 preserves the useful course material and records its limitations. Reusable prediction code will be added only after approval of the relevant stages. The final recruiter-facing README is reserved for Stage 8.

## Current layout

```text
NBA/
├── .gitignore
├── README.md
├── requirements.txt
├── data/raw/
│   ├── README.md
│   ├── nba_injury_logs.csv                         [local, ignored]
│   ├── nba_player_season_stats.csv                 [local, ignored]
│   ├── nba_player_box_scores.csv                   [local, ignored]
│   └── supplementary_season_stats_2009_2023.zip    [local, ignored]
├── notebooks/legacy/
│   ├── nba_course_early.ipynb
│   └── nba_course_final.ipynb
├── src/legacy/nba_course.py
├── scripts/verify_migration.py
├── results/legacy/
│   ├── README.md
│   ├── figures/                                   [11 saved PNGs]
│   ├── figure_manifest.json
│   ├── notebook_text_outputs.json
│   └── saved_model_metrics.json
└── docs/
    ├── original_project_review.md
    ├── repository_plan.md
    ├── migration_manifest.json
    ├── data_profiles.json
    ├── additional_findings.json
    ├── supplementary_archive_profiles.json
    ├── migration_verification.json
    └── legacy/                                    [report/slides local, ignored]
```

The local Git metadata is omitted from the tree. No unrelated assignments, duplicate primary ZIPs, macOS metadata, or course instruction files were copied.

## Add as later stages are approved

| Path | Role | Stage |
| --- | --- | --- |
| `src/nba_injury/data/` | Loading, identity resolution, event cleaning, eligible observations, temporal splits | 2 |
| `docs/prediction_design.md` | Final task and leakage protections | 2 |
| `data/processed/` | Generated observation tables and split membership, ignored by Git | 2–3 |
| `src/nba_injury/features/` | Strictly historical workload, performance, and injury-history features | 3 |
| `docs/feature_dictionary.md` | Features, units, windows, availability rules | 3 |
| `notebooks/` | Portable explanatory EDA and evaluation notebooks | 3 onward |
| `src/nba_injury/models/` | Training, imbalance handling, tuning, frozen model selection | 4–5 |
| `src/nba_injury/evaluation/` | Metrics, ROC/PR curves, calibration, interpretation, applied risk plots | 4–7 |
| `results/figures/` and `results/tables/` | Verified portfolio results with provenance | 3–7 |
| `tests/` | Focused temporal-leakage and data-integrity checks where needed | 2 onward |

Do not pre-create empty modules or implement future work in Stage 1. Keep the original material unchanged in `courses/585`; review retention of the local legacy copies during Stage 8 cleanup.
