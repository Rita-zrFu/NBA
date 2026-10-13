# Local source data

These files are byte-for-byte copies of the course material. They are ignored by Git pending source-license and redistribution checks. No data license is inferred from public availability. The migration manifest records original filenames, sizes, and SHA-256 hashes.

| Local filename | Records | Meaning |
| --- | ---: | --- |
| `nba_injury_logs.csv` | 37,667 | Dated availability/injury-list transactions, 1951-12-25 to 2023-04-16 |
| `nba_player_season_stats.csv` | 28,237 | Player/team/season totals and age/position, numeric years 1950–2022, including traded-player aggregate and team rows |
| `nba_player_box_scores.csv` | 1,309,778 | Player/game box scores, 1946-11-01 to 2023-04-05 |
| `supplementary_season_stats_2009_2023.zip` | 8,980 across 14 CSVs | Per-game season averages for 2009–10 through 2022–23 and `Player-additional` identifiers |

The report attributes the primary inputs to Logan Lauton's Kaggle [NBA Injury Stats](https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023) and [NBA Players & Team Data](https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data). The pages resolve, but their license/content details were not readable through the inspection browser. The supplementary archive's upstream provenance is not documented in the course files.

The primary box-score `Season` field equals the calendar year of `GAME_DATE` for every row. Do not assume it identifies an NBA season. The primary season table has totals, while the supplementary tables have per-game averages. Do not combine their numeric values without unit checks.

These sources contain no height, weight, biometrics, injury severity, confirmed onset times, or official prospective at-risk roster. Historical games alone cannot establish a complete roster or reason for every absence. The original date coverage is partial in 2023.
