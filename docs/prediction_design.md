# Stage 2 prediction design and feasibility

Stage 2 is complete. The data support a restricted benchmark for future **documented injury-related inactive-list episodes** after recent NBA appearances. They do **not** establish a clean, continuously observed NBA at-risk roster, confirmed injury onset, or an unbiased population injury-risk dataset. The benchmark below is the strongest supported alternative for review. It should proceed to modeling only if this narrower interpretation and its ascertainment limitations are acceptable.

No model, full feature pipeline, or preprocessing transformation was fitted. All original course files and migrated source copies remain unchanged.

## Proposed task

At an eligible player-date snapshot, use strictly historical, delayed source information to predict at least one qualifying new documented injury-related inactive-list episode within the next **14 calendar days**. Preserve a 7-day alternative. The endpoint is an administrative recorded transition following evidence of active status, not a new physiological injury or a medical diagnosis.

| Design element | Definition |
| --- | --- |
| Unit | Appearance-anchored player-date snapshot, with at least 14 days between snapshots for the same player |
| Anchor | A box-score row with positive recorded minutes and an unambiguous player identity |
| Index date | Anchor game date plus two calendar days |
| Historical feature cutoff | Exclusive index date minus one calendar day; all feature-source dates must be strictly earlier |
| Example | Anchor November 14 → index November 16 → source dates through November 14 → 14-day outcome interval November 16–29 |
| Primary horizon | Index date through index date + 13 days, inclusive |
| Alternative horizon | Index date through index date + 6 days, inclusive |
| Minimum history | Ten positive-minute appearances available within the observed schedule block, including the anchor, and 30 calendar days of block history before the feature cutoff |
| Primary population | Recently observed regular-season NBA participants with adequate history and labelable follow-up |
| Intended evaluation | Future chronological periods, including returning players and some players not seen in training |

The extra calendar day between the anchor and index avoids assuming that a late game's complete statistics were available at midnight immediately afterward. Intraday game, transaction, and publication timestamps are absent. The buffer is a conservative recorded-date policy, not a certification of actual publication time or later dataset revisions.

## Why this observation unit

| Candidate unit | Advantages | Limitation in these inputs | Decision |
| --- | --- | --- | --- |
| Player-game, before that game's start | Natural basketball decision point and prior workload | Actual game appearance is learned later; starting times and scheduled player availability are unavailable | Avoid retrospective selection of a supposedly pregame cohort |
| Every player/calendar date | Full daily trajectory | No dated roster, contracts, transfers, or assignment status to justify creating daily active-player rows | Do not manufacture rows for absent or departed players |
| Fixed weekly snapshot | Fewer observations and calendar interpretation | Activity evidence may be several days old; complete weekly rosters are still unavailable, and 14-day labels overlap | Not selected |
| Appearance-anchored, spaced player-date | Prior participation is observed, features can be dated strictly before the index, and labels need not overlap | Restricts the population to regularly appearing players; current roster status remains a proxy | Selected for the restricted benchmark |

Sampling is deterministic and chronological. It takes the first eligible anchor, then the next eligible index at least 14 days later. Future target values do not determine this spacing. If a selected row later has an unknown label, the sampling clock is not reset to hunt for another labelable row. The 7-day and 14-day alternatives start from the same snapshots.

## Outcome construction and inactive episodes

The source is an availability/transaction log. `Relinquished` plus an injury word is insufficient to establish onset. The rule order is explicit in `src/nba_injury/data/design.py`.

1. Remove identical source records after excluding CSV export-index columns. Two duplicate rows are removed from the complete source input.
2. Treat explicit activations and returns to the lineup as returns. A misplaced-column activation is interpreted by its note, not treated as an injury placement. A return-to-team phrase alone does not establish playing status.
3. A positive target requires an explicit `placed on IL`, `placed on IR`, `placed on inactive list`, or `placed on injured list` action with a relinquished player. A DTD/DNP symptom note without the placement action is excluded from positive outcomes.
4. Include explicit injury-related conditions such as sprains, strains, fractures, contusions, concussion, tendon conditions, anatomically described pain/soreness, and specified musculoskeletal/traumatic conditions. These are descriptions, not clinically adjudicated diagnoses. Mixed injury and illness/protocol/rest descriptions are ambiguous.
5. Exclude activations, rest/load management, protocols, illness and other nonmusculoskeletal medical conditions, personal/disciplinary reasons, and conditioning/ineligibility from the main positive outcome. These placements still interrupt documented active status. A later injury placement can qualify after a documented return or subsequent appearance.
6. Recovery, surgery, unspecified reasons, imprecise dates, and nonplacement status notes are uncertain for an incident target when the player was active or the prior state was unknown. Their affected horizons are censored rather than converted into negatives.
7. A placement is new only when the strictly preceding recorded state is active, established by an activation/return-to-lineup or a positive-minute appearance on an earlier date. An initial placement without prior active evidence is unconfirmed.
8. Subsequent inactive-to-inactive updates belong to the existing episode. They do not create another positive or invalidate a clearly observed incident solely because the record was repeated. A subsequent recorded active return or appearance can open eligibility for another episode.
9. Multiple events on one player/date are collapsed. Conflicting reasons or entry and return on the same date are unordered and uncertain. An appearance on the placement date never establishes the required pre-event active state.

Within the chosen overall calendar span, 2,648 notes were classified as injury-related; 2,646 were matched to a usable identity. After transition reconstruction, the span contains 2,568 confirmed documented incident injury-related episodes across all matched participants, before observation eligibility and horizon sampling. These are source-event counts, not the final number of positive snapshots.

For any horizon, uncertainty anywhere in the window excludes both positive and negative observations. A known event earlier in a globally unresolved window is not selectively retained. Known repeats are handled separately from uncertainty. A label of zero means no qualifying documented target in the retained window, not proven absence of injury.

Full reason inventories and row-level identity audits are retained locally. Aggregate counts are in `results/tables/stage2_note_rule_counts.csv`, `stage2_episode_audit.csv`, and `stage2_reporting_by_phase.csv`.

## Identity resolution and coverage

The primary sources have names but no stable player identifier. Normalize Unicode accents, case, spacing, and punctuation. Inspect slash-separated aliases separately. Retain suffix distinctions whenever possible. A missing suffix is resolved only when the candidate base name is unique. Ambiguous matches remain unresolved; no fuzzy matching is used.

Three reviewed corrections are explicit: `D.J. Augustine` → `D.J. Augustin`, `Roy Devyn Marble / Roy Marble (Devyn)` → `Devyn Marble`, and `Kenyon Martin Sr.` → `Kenyon Martin`. The last distinguishes the older player from `Kenyon Martin Jr.`. For these aliases, all 60 inspected log rows have an appearance on the matching team within 56 days of the log date. The evidence is retained locally. Bare `Roy Marble` is not automatically mapped to Devyn.

The supplementary archive exposes multiple `Player-additional` identifiers under each of four primary box-score names: Chris Johnson, Chris Wright, Mike James, and Tony Mitchell. All four names are excluded from the cohort. A modern Chris Johnson game has two team rows under the same name/game ID, directly illustrating the problem. Homonym evidence is saved in `stage2_homonym_evidence.csv`.

| Coverage scope | Matched rows / actor rows | Coverage |
| --- | ---: | ---: |
| All audited dates, January 2009–April 2023 | 24,323 / 24,409 | 99.6477% |
| Chosen calendar span, October 16, 2018–April 10, 2022 | 7,818 / 7,832 | 99.8212% |
| Relinquished actor rows in the chosen span | 4,136 / 4,143 | 99.8310% |
| Injury-related note rows in the chosen span | 2,646 / 2,648 | 99.9245% |

These denominators count the actor selected by the note/action once per deduplicated source row. They are matching coverage, not independently established correctness. In the chosen span, 12 matched rows use a reviewed alias, 93 use a unique suffix fallback, and 7,713 use normalized exact/slash-alias matches.

Unresolved names include never-observed participants, a team name in the player field, dates in the name field, and diagnostic phrases such as `left wrist injury (DTD)`. No missing or unmatched actor is quietly treated as a negative. Unresolved outgoing-event dates quarantine **all league-wide candidate windows** that cover the date, including both classes. This avoids assuming an unknown player belongs to the anchor team or ignoring an unobserved transfer. Homonymous identities already excluded from the whole cohort do not trigger quarantine for other players. Unmatched activations cannot restore a player's state; a later observed appearance can. A later unconfirmed injury transition is censored.

## At-risk eligibility and follow-up

Each candidate uses a recently observed positive-minute game, the minimum history above, and no outgoing/uncertain status entry for that player on the anchor date or the intervening date before index. Globally unresolved outgoing-event dates in this interval also exclude anchors. This avoids knowingly indexing an already inactive player and does not create synthetic days for players who stopped appearing.

The full maximum horizon must fit within one observed schedule block. Blocks are identified from league game dates, with gaps over 45 days separated. No window crosses an offseason or the 141-day 2020 interruption. The short 2020 restart block cannot meet the 30-day history criterion. The partial 2022–23 block is excluded entirely.

For a negative label, additionally require a matched NBA observation on/after the horizon end within seven days, still inside the same block. A positive-minute appearance, explicit activation, or explicit placement can serve as this follow-up witness. Terminal/no-observation windows are unknown, not negative. Positive events are observable once recorded and do not need a later return to play to establish that an event occurred. This retains long injuries but creates outcome-dependent ascertainment: known events and incomplete negative follow-up have different observability.

**A clean, continuously observed at-risk cohort cannot be verified from these files.** A near-endpoint observation does not prove uninterrupted roster membership. Unreported departures, G League assignments, intervening unreported injuries, and source revisions remain possible. Follow-up filtering and minimum appearance history introduce selection and informative censoring. The proposed classification benchmark applies to this restricted, retrospectively labelable population, not all players active at prediction time. A general injury-risk cohort would require dated roster/contract/assignment records, consistently captured availability reports, and better event/publication timestamps.

## Sample sizes and horizons

There are 97,235 positive-minute appearance anchors in the candidate blocks after the homonym restriction. The deterministic eligibility and spacing rules retain 11,511 snapshots before horizon-specific label ascertainment.

| Horizon | Labelable observations | Positive | Negative | Prevalence | Unique players |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 days | 10,637 | 750 | 9,887 | 7.0509% | 707 |
| 14 days, primary | 9,804 | 1,331 | 8,473 | 13.5761% | 691 |

For 14 days, 1,138 selected snapshots are censored by ambiguous/unresolved forward events and 569 by missing nearby forward observations. For 7 days the respective counts are 366 and 508. They are not imputed as zero. The primary negative-to-positive ratio is approximately 6.37:1, so class imbalance is material. Training prevalence, rather than combined/test prevalence, must determine future imbalance-handling parameters.

The two horizons have different labelability masks. On the **same 9,517 rows** with both labels available, 7 days has 701 positives (7.3658%) and 14 days has 1,331 (13.9855%). The longer horizon captures more documented events on identical snapshots and provides more events for later modeling. It is chosen for that task-level reason; no model performance was used to select the horizon. The alternative remains available and must not be turned into a new chance to tune on the test set.

## Chronological train, validation, and test design

Use the later reporting era beginning with 2018–19. Earlier blocks frequently have roughly half of outgoing entries without a usable incident reason. Uncertain/recovery fractions are 30.27% in 2018–19, 17.28% before the 2020 interruption, 6.17% in 2020–21, and 11.35% in 2021–22. Starting later reduces, but does not remove, reporting drift.

Training uses 2018–19 and the observed pre-interruption portion of 2019–20. Validation uses 2020–21. Test uses 2021–22, the latest fully represented regular season: 1,230 distinct games and 82 games for each of 30 teams, October 19, 2021–April 10, 2022. The latest input block has 1,196 games ending April 5, 2023 and is not used as the final test season.

| Split | Actual index-date range, primary 14 days | Observations | Positive | Prevalence | Players |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | 2018-11-16–2020-02-27, in separated regular-season blocks | 4,957 | 593 | 11.9629% | 521 |
| Validation | 2021-01-22–2021-04-23 | 2,143 | 293 | 13.6724% | 450 |
| Test | 2021-11-19–2022-03-28 | 2,704 | 445 | 16.4571% | 462 |

The latest horizon ends are March 11, 2020, May 6, 2021, and April 10, 2022, respectively. Negative follow-up witnesses may occur later than the horizon end but remain within the source block. Applying the same two-day recorded-date availability policy, the latest assumed label-availability dates are March 13, 2020, May 13, 2021, and April 12, 2022. Training labels are thus available before validation indices, and validation labels before test indices.

| Split | 7-day observations | Positive | Prevalence | Players | Actual index dates |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 5,317 | 312 | 5.8680% | 523 | 2018-11-16–2020-02-27 |
| Validation | 2,376 | 184 | 7.7441% | 465 | 2021-01-22–2021-04-30 |
| Test | 2,944 | 254 | 8.6277% | 470 | 2021-11-19–2022-03-28 |

No shuffled split is used. Model/test metrics do not exist yet. Stage 2 inspected test outcome counts to audit cohort design, as requested; no predictions, hyperparameters, or feature sets were selected using test performance.

## Repeated measurements and generalization

Primary observations per player: minimum 1, median 13, mean 14.19, 90th percentile 27, maximum 31. Within train/validation/test, mean observations are approximately 9.51/4.76/5.85; medians are 10/5/6; maxima are 16/7/9. Future outcomes do not overlap between successive snapshots for the same player, and one documented incident is never counted as the first target for two observations in the same horizon dataset.

Player intersections in the 14-day cohorts: train/validation 359, train/test 314, validation/test 369, all three 300. Of 462 test players, 314 were in training and 148 were not. This is evaluation at future time points with mixed returning/new players, not an experiment restricted to unseen players. Shared identities across chronological periods are permitted for this goal. Repeated observations remain dependent; ordinary row-based independence assumptions and row-level confidence intervals would be inappropriate.

## Historical feature feasibility, without engineering the pipeline

| Feature family | Available evidence | Safe timing / qualification |
| --- | --- | --- |
| Minutes and games played | Dated box scores, numeric `MIN` | Count/sum positive-minute appearances strictly before `history_cutoff_exclusive`; zeros cannot automatically identify DNPs |
| Recent/cumulative workload | Game dates and minutes | 7/14/30-day windows and observed-block cumulative counts/totals are feasible, ending at the delayed cutoff |
| Rest and back-to-back workload | Consecutive observed appearance dates | Gaps and consecutive-day appearances are feasible; describe them relative to the last available game, not as verified training/rest/travel time |
| Recent performance | `PTS`, shooting makes/attempts and percentages, rebounds, assists, steals, blocks, turnovers, fouls, plus/minus | Strictly historical windows; check ratios/units and missing values in Stage 3 |
| Game-level starts | Absent from box-score columns | Cannot create rolling starts. `GS` only exists in the full-season table and must not be used from the current unfinished season |
| Prior injury counts / days since injury | Matched, qualifying episode dates | Count only earlier confirmed recorded episodes before the delayed cutoff; left-truncated, reporting-dependent history, not lifetime clinical history |
| Age, position, prior-season totals | Season table | Potentially usable after validating season conventions, traded-player duplicates, and lagging completed seasons. Not approved as current-season predictors here |
| Height, weight, BMI, biometrics, severity, true missed days | Unavailable or not reliably identified | Do not invent them or infer missed games/days from every absence |

No rolling workload/performance columns, feature dictionary, encoders, imputers, scaling, correlations, or models are implemented in Stage 2. `prior_appearances_in_phase` is eligibility bookkeeping only.

## Leakage protections and future modeling constraints

- No anchor or feature source date reaches the index date; the feature cutoff is additionally delayed by one day.
- Episode construction processes history forward. Future returns/appearances cannot change the classification of an earlier episode. Same-day events cannot establish ordering.
- Sampling and spacing are decided before labels. Unknown outcomes remain missing. Global source-quality exclusions are applied to both classes.
- Labels and follow-up witnesses are used only for outcome ascertainment. Never use targets, label statuses, outcome dates, witness dates, assumed label-availability dates, horizon ends, inferred phase ends/IDs, or split membership as predictors.
- Training/validation/test are chronological. All feature transformations, selection, weights, tuning, and any calibration must be learned using the permitted training/validation data only.
- Future temporal CV must purge by `assumed_label_available_date`, not merely split on index dates. A 14-day label may need a witness up to seven days later, plus the two-day availability policy: up to 22 days after index. A 23-day boundary separation is a conservative fallback. CV is not implemented or tuned in Stage 2.
- Earlier-period raw games and events can remain legitimate history for later snapshots if their source dates precede that row's cutoff. Current/full-season aggregates and future episodes cannot be used.
- Identity normalization is fixed metadata resolution, not outcome-based encoding. Names/IDs used for grouping should not become player memorization features by default.

## Remaining limitations and review decision

The principal unresolved issue is cohort ascertainment. We have recent participation evidence and conservative exclusions, not an official complete roster or certified uninterrupted follow-up. Complete-case selection can favor established players and observable outcomes; apparent prevalence and later calibration are conditional on that selection. A model trained here cannot automatically provide unbiased probabilities for all NBA players.

Reporting quality and endpoint prevalence change over time. Symptoms may precede recorded placement, records may be revised, and a new administrative episode may represent recurrence of the same physical injury. High name-match coverage cannot establish ground-truth identity correctness without shared stable IDs. The sources lack postseason games, intraday/ingestion timestamps, true medical onset, severity, consistent missed-day information, and verified reasons for every absence. These limitations cannot be repaired by model tuning.

Stage 2 therefore stops with a **restricted, conditional recorded-event design**, its reproducible candidate cohorts, and the limitations above. A strict clean at-risk injury cohort would need additional data. No broad injury-risk claim is established, and no Stage 3 work should start until the user explicitly accepts a task to continue with.

## Files and reproduction

Reusable code: `src/nba_injury/data/design.py`, package initializers, and `scripts/build_stage2_design.py`. Rules are frozen in `docs/stage2_design_config.json`. Timing/identity/episode tests are in `tests/test_stage2_design.py`.

The builder writes candidate observations, 7/14-day labelable cohorts, train/validation/test files, status records, and episode records to `data/processed/stage2/`, which remains ignored by Git. Aggregate audits are in `results/tables/stage2_*.csv` and `stage2_design_summary.json`. Row-level source-note/name inventories are also retained locally and ignored. `.gitignore` and the preliminary README are updated. The Stage 1 source review, copied course files, legacy figures, and requirements remain unchanged.

From the project root, with the original migrated data in place:

```bash
python3 scripts/verify_migration.py
python3 scripts/build_stage2_design.py
python3 -m unittest discover -s tests -v
git status --short
```

The final builder and all eight tests ran successfully using existing Python 3.13.7, pandas 2.3.2, and NumPy 2.3.3. No package was installed, no unrelated environment was modified, and no locked project environment has been created yet. Source hashes match the Stage 1 manifest. No model training, full feature engineering, Git staging, commit, remote mutation, or push occurred.
