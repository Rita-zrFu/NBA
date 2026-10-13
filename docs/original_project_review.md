# Stage 1 review of the original NBA project

Inspection completed on October 13, 2026. The source directory is the user's `courses/585` course folder. The new project lives separately in `Internship/NBA`. Stage 1 covers inspection, migration, and a repository plan. No new prediction task, temporal split, feature pipeline, or models have been implemented.

## What was found

The NBA work is concentrated in `final proj/`, with an earlier notebook, an additional season-data archive, and the final written report in the course folder itself. The remaining Python assignments, reports, and the course `figures/` directory concern unrelated exercises and were excluded. No NBA README, requirements file, environment file, lock file, standalone saved model, or reusable pipeline was found.

| Original source | Observed role | Retained destination |
| --- | --- | --- |
| `final proj/nba.ipynb` | Final notebook, 12 code cells, 11 embedded plots, saved printed results | `notebooks/legacy/nba_course_final.ipynb` |
| `nba.ipynb` | Earlier notebook, 11 code cells and 10 plots, same saved logistic regression result | `notebooks/legacy/nba_course_early.ipynb` |
| `final proj/nba.py` | Script export with English comments, same main analysis and model | `src/legacy/nba_course.py` |
| `final proj/NBA Player Injury Stats(1951 - 2023).csv` | Primary availability/injury log | `data/raw/nba_injury_logs.csv` |
| `final proj/NBA Player Stats(1950 - 2022).csv` | Primary season statistics | `data/raw/nba_player_season_stats.csv` |
| `final proj/NBA Player Box Score Stats(1950 - 2022).csv` | Primary game statistics | `data/raw/nba_player_box_scores.csv` |
| `nba player data 09-23.zip` | Supplementary season averages, includes player identifiers and 2022–23 | `data/raw/supplementary_season_stats_2009_2023.zip` |
| `NBA_Final_Report_RitaFu.docx` | Final written course report | `docs/legacy/NBA_Final_Report_RitaFu.docx` |
| `final proj/final_presentation_RitaFu.pptx` | 14-slide course presentation | `docs/legacy/final_presentation_RitaFu.pptx` |

The final notebook includes a 76ers time-series plot absent from the script. The earlier notebook lacks the final notebook's pre-injury comparison section. Both notebooks retain the same saved model metrics. All retained source files are unchanged copies, including their original paths, comments, outputs, and kernel metadata. Their known defects are documented rather than silently corrected.

Duplicate compressed versions of the three primary datasets and the extra injury CSV in the course root were not copied. The supplementary ZIP is distinct and was retained because it adds identifiers, different units, and 2022–23 data. It was inspected but was not used by the original NBA code, and it is not yet incorporated into a new pipeline. macOS metadata, course guidelines, unrelated assignments, and unrelated plots were excluded.

## Data sources and actual coverage

The report cites Logan Lauton's Kaggle [NBA Injury Stats](https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023) and [NBA Players & Team Data](https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data). These URLs resolve, but detailed upstream content and licensing could not be read through the inspection browser. Source attribution comes from the course report. The supplementary archive has no documented upstream citation. No redistribution permission is inferred. Raw data and original Office documents remain ignored by Git.

| Data | Observations | Fields excluding export indices | Actual coverage | Unit |
| --- | ---: | ---: | --- | --- |
| Injury log | 37,667 | 5 | 1951-12-25 to 2023-04-16 | Dated transaction/availability-log entry |
| Season statistics | 28,237 | 30 | Numeric `Season` 1950–2022 | Player/team/season row, including `TOT` aggregate rows |
| Game box scores | 1,309,778 | 28 | 1946-11-01 to 2023-04-05 | Player/game row |
| Supplementary ZIP | 8,980 across 14 CSVs | 31 per table, including `Rk` | Filenames 2009–10 through 2022–23 | Player/team/season averages |

The filenames understate actual box-score coverage. Both primary date columns parse without invalid dates. Primary box-score `Season` equals the calendar year of `GAME_DATE` in every row. The season-statistics year is a numeric field, without a start/end date or explicit NBA season string. The course helper's comment about parsing `2015-16` does not describe these actual numeric inputs. NBA season alignment must be checked before integration.

The primary season table contains season totals: for example, its 2022 Precious Achiuwa row has `G=73`, `MP=1725`, and `PTS=664`. The supplementary archive contains per-game averages, for example `MP=8.5` in a 2010–11 player row. The report's references to season-table minutes per game do not match the primary data's units.

The raw log has columns `Date`, `Team`, `Acquired`, `Relinquished`, and `Notes`. Common notes include 17,434 occurrences of `activated from IL`, 6,331 of `placed on IL`, 522 of `placed on IL with NBA health and safety protocols`, and 215 of `placed on IL for rest`. This is broader than a collection of confirmed new injuries. `Relinquished` is missing in 17,623 rows and `Acquired` in 20,044 rows, largely reflecting transaction direction. There are no stable player IDs in the primary inputs. The supplementary `Player-additional` field may help identity resolution but has not been validated or joined.

Machine-readable columns, data types, missing values, samples, year counts, and duplicate checks are in `data_profiles.json`. Archive schemas are in `supplementary_archive_profiles.json`. Additional inspection findings are in `additional_findings.json`.

## Original objective and analysis

The course project described NBA injury patterns and explored whether season statistics could classify players with an injury-log record. It asked about changes over time, injury types by position, differences between teams, changes before an injury entry, and season-level logistic regression. It did not define a prospective prediction time or horizon.

The code removes CSV export-index columns, parses dates, uppercases and strips player names, extracts numeric years, and classifies free-text notes by first matching keyword. The seven categories are Ankle/Foot, Knee/Leg, Back, Shoulder/Arm, Head/Neck, Illness, and Torso/Core. It saves 12,038 classified records in the final notebook output. Uppercase/strip is the actual name cleaning. The report's description of punctuation removal and the slides' name-format examples exceed the implemented normalization.

The analysis uses multiple observation units:

- **Log entry:** counts and keyword-based injury types.
- **Calendar year, decade, position, or team/year:** aggregated descriptive summaries.
- **Selected injury-log entry:** last five games versus earlier games before that entry, with at least ten prior games in the matched calendar year.
- **Player/team/season row:** the logistic regression input, rather than one distinct row per player/season.

## Existing features and outcomes

| Use | Variables or derived quantities | Implementation |
| --- | --- | --- |
| Logistic regression inputs | `Age`, `MP`, `G`, `GS`, `PTS`, `FGA`, `TRB`, `STL`, `BLK`, `PF`, `TOV` | Eleven full-season features, complete-case filtering, standardization fitted on training data |
| Exploratory season features | Above plus `AST` | Histograms and correlation heatmap for numeric years 2002–2022 |
| Descriptive groups | Parsed date/year, decade, standardized name, `injury_group`, simplified `Pos` G/F/C, `Team` | Aggregation and plots |
| Pre-injury comparison | Last-five-game and earlier-game means for `MIN`, `FG_PCT`, `PTS`; their differences | First 500 chronologically ordered classified entries in 2010–2022, 319 retained with enough matched games |
| Original model label | `y_injured` | One if a normalized `Relinquished` name has any raw log entry in the matching calendar year, otherwise zero |

The model label uses the **unfiltered** injury log, unlike the descriptive analysis's keyword-classified subset. Grouping ignores missing `Relinquished` values, so activation rows with no relinquished player generally do not contribute a player label. Generic inactive-list placements, rest entries, and unmatched-name assumptions still affect the label. No record becomes zero even if the reason is incomplete reporting or failed name matching. This is an observed log-based proxy with uncertain meaning.

The pre-injury deltas are calculated for descriptive analysis only. They are not added to the logistic regression features. No rolling day windows, rest-day features, cumulative prior injury counts, days since injury, or reusable injury-history pipeline exists in the course code. Height, weight, BMI, explicit experience, direct usage rate, medical data, and confirmed injury onset are unavailable in the inspected inputs.

## Methods and saved results

The existing methods are descriptive aggregation, plots, and one `LogisticRegression(max_iter=2000)`. The split is a random stratified 70%/30% train/test split with `random_state=42`. `StandardScaler` is correctly fitted on the training rows only. The original code uses no cross-validation, separate validation set, tuning, class weights, tree models, PR-AUC, calibration, permutation importance, or SHAP.

The following results are **read from saved notebook outputs, not newly trained or prospective evaluations**:

| Quantity | Saved result |
| --- | ---: |
| Original model rows | 12,711 |
| Positive rows | 6,978 (54.8973%) |
| Negative rows | 5,733 |
| Training/test rows | 8,897 / 3,814 |
| Exact ROC-AUC | 0.5362580795628706 |
| Accuracy, rounded | 0.553 |
| Positive precision, rounded | 0.558 |
| Positive recall, rounded | 0.895 |
| Positive F1, rounded | 0.688 |
| PR-AUC / average precision | Not reported |

Inspection of the original merge reproduces the 12,711 rows and 6,978 labels **without fitting a model**. There are only 10,103 distinct normalized player/year keys, with 3,868 rows belonging to repeated keys in the 2002–2022 modeling range. All eleven original features are complete in that range. Positive prevalence is close to 55%, so this original label is not strongly imbalanced. Imbalance must be measured again after a defensible prospective task is defined.

The saved pre-injury comparison includes 319 entries and average last-five-minus-earlier differences of −1.459707 minutes, −0.013818 field-goal proportion, and −0.869537 points. This selected case-only comparison has no non-injury control windows and does not establish predictive usefulness.

The eleven embedded plots were extracted unchanged into `results/legacy/figures/`: season distributions, yearly log counts, correlations, decade injury groups, position proportions, mean team counts, Blazers counts, 76ers counts, pre-injury deltas, logistic coefficients, and the ROC curve. Saved printed tables and model metrics are preserved beside the figures. No separate exported NBA figure directory or persisted model was found in the source.

## Dependencies and reproducibility

Observed scientific imports are pandas, NumPy, Matplotlib, Seaborn, and scikit-learn. Notebook metadata identifies Python 3.13.5 and a Conda base kernel. No package versions are recorded, and the metadata is not an environment specification. `requirements.txt` declares these unpinned imports plus JupyterLab for notebook use. No packages or project environment were installed in Stage 1.

The legacy script and notebooks refer to an absolute `COURSE/585/final proj` path rather than the actual inspected `courses/585` directory. They use in-memory state and `plt.show()` rather than portable data/output paths. The final notebook's execution counts are not sequential, although no saved error outputs were found. The copies are deliberately unchanged and should be treated as historical references until the later refactor/cleanup stages.

## Major limitations requiring work

1. **Prediction timing and leakage:** full-season totals can include information after the events being classified. The original project has no prediction point or horizon. Injury-related missed games can also affect these predictors.
2. **Year alignment:** calendar-year injury dates are joined to a season table without validated NBA season mapping. Box-score `Season` is demonstrably calendar year. This can misassign histories and outcomes across seasons.
3. **Repeated rows and split contamination:** traded-player aggregate/team rows are all retained. A random row split can place the same player/year in training and test, while future years can inform earlier evaluations.
4. **Unreliable outcome proxy:** inactive-list placements, illness, rest, unspecified notes, activations, and repeated status changes require explicit handling. Entry date is not necessarily injury onset. Missing records and failed name matches are not verified absence of injury.
5. **Identity matching:** uppercase/strip cannot resolve aliases, suffixes, punctuation, accents, or multiple names. There are 1,061 relinquished-name rows containing ` / ` aliases. `Game_ID` should be loaded as a string in future code to preserve leading zeros. It is a game identifier, not a player identifier.
6. **Duplicates and observation integrity:** across all raw years, 7,734 season rows belong to repeated player/season keys, and 148 game rows belong to repeated player-name/game keys. These are not all identical duplicates and require contextual investigation. There are two identical injury records after excluding the CSV index column. Nothing was deduplicated in Stage 1.
7. **Historical missingness and availability:** older box scores lack many statistics. The 2002–2022 calendar-year subset has 523,070 rows and only 81 missing `REB` values among the inspected columns, but includes 2,223 zero-minute rows. An appearance table alone does not provide a complete at-risk roster or explain every absence. The 2023 coverage is partial and the injury log extends eleven days past the final box-score date.
8. **Misleading descriptive labels:** the decade denominator is distinct players, despite the axis/report describing player-seasons. Team plots filter 2015–2022 despite titles saying 2000–2022. Counts of log entries are not confirmed incidence rates, and changes in reporting can affect trends. Team/year combinations with no recorded events are omitted from the mean rather than explicitly assigned zero.
9. **Selection and interpretation:** only the first 500 sorted classified events enter the pre-injury analysis, only 319 qualify, and there are no control windows. No uncertainty or hypothesis-testing analysis supports claims that the declines are meaningful. The report's causal suggestions about play style, pandemic schedules, and team factors are not established by the code.
10. **Evaluation and résumé gaps:** the ROC-AUC is near chance, no prospective test design exists, and the random split does not support the proposed résumé's CV, model comparison, imbalance handling, or tuning claims. Performance improvement cannot be promised from the available data.

## Additional ML work needed after approval

Stage 2 must first validate event direction/meaning, identity resolution, season convention, observation eligibility, follow-up completeness, and the availability of each variable at prediction time. It will define a meaningful prediction unit and horizon and save a defensible temporal train/validation/test design. Dated game records make historical workload and repeated evaluation observations plausible, but feasibility and coverage must be established before committing to a longitudinal task.

Stage 3 will build strictly historical workload/performance windows and injury-history features, with a feature dictionary and training-only preprocessing. Stages 4–5 will compare logistic regression, random forest, XGBoost, and LightGBM if feasible, using PR-AUC and ROC-AUC, consistent imbalance handling, temporally appropriate cross-validation, tuning, and a frozen final test evaluation. Stages 6–7 add calibration, interpretation, and applied player risk plots if justified. Stage 8 produces the final portfolio README and portable repository cleanup.

Every résumé statement must be checked against implemented code and verified results. The original project currently supports Python data integration, descriptive analysis, and a simple logistic regression only.

## Migration verification and review boundary

Nine source files totaling 174,431,359 bytes were copied. Source and destination SHA-256 hashes match. All 55 original course files were compared with hashes captured before migration: no file contents changed, no source files were added, and no source files were removed. Exact copy mappings and hashes are in `migration_manifest.json`, with verification recorded in `migration_verification.json`.

The verifier checks retained files and historical figures using Python's standard library:

```bash
python3 scripts/verify_migration.py
python3 scripts/verify_migration.py --source /path/to/courses/585
git status --short
```

Only a local Git repository is initialized. No files are staged, no commits or remotes are created, and no push, pull request, or remote mutation is performed. Stage 1 ends here. Stage 2 requires the user's explicit instruction to continue.
