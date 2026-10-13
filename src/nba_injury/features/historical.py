"""Build fixed-schema features from dated source history, never from labels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOWS = (7, 14, 30)
INJURY_WINDOWS = (30, 90, 180, 365)
PERFORMANCE_MEANS = {
    "pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL",
    "blk": "BLK", "tov": "TOV", "pf": "PF", "plus_minus": "PLUS_MINUS",
    "fga": "FGA", "fg3a": "FG3A", "fta": "FTA",
}
RATIOS = {"fg": ("FGM", "FGA"), "fg3": ("FG3M", "FG3A"), "ft": ("FTM", "FTA")}
SNAPSHOT_FIELDS = ["observation_id", "player", "index_date", "history_cutoff_exclusive", "phase_start"]


def feature_dictionary():
    records = []
    def add(name, group, description, source, units, window=None, retained=True):
        records.append(dict(feature=name, group=group, description=description, source=source,
            timing_rule="historical game/episode date < row history_cutoff_exclusive; index date is a known clock reference",
            window_rule=window or "all observed history in the stated scope", units=units, retained=retained))
    for w in WINDOWS:
        window=f"[cutoff - {w} days, cutoff), within current observed schedule block"
        add(f"appearances_{w}d", "workload", "Count of positive-minute appearances", "dated box scores", "appearances", window)
        add(f"minutes_sum_{w}d", "workload", "Total recorded minutes across appearances", "MIN", "minutes", window)
        add(f"minutes_mean_{w}d", "workload", "Average recorded minutes per appearance", "MIN", "minutes/appearance", window)
        add(f"back_to_back_pairs_{w}d", "workload", "Consecutive appearance-date pairs exactly one day apart; both games inside window", "GAME_DATE", "pairs", window)
    add("block_appearances", "workload", "Available appearances since the observed block began", "GAME_DATE, MIN", "appearances", "[phase_start, cutoff)")
    add("block_minutes", "workload", "Available minutes since the observed block began", "MIN", "minutes", "[phase_start, cutoff)")
    add("days_since_available_appearance", "workload", "Index date minus last available appearance date; constant 2 under frozen anchoring, excluded from X", "GAME_DATE, index_date", "days", retained=False)
    for name, column in PERFORMANCE_MEANS.items():
        add(f"{name}_mean_30d", "performance", f"Mean observed {column} per appearance; missing source values are not zero-filled", column, "statistic/appearance", "[cutoff - 30 days, cutoff), within block")
    for name, (made, attempted) in RATIOS.items():
        add(f"{name}_pct_30d", "performance", f"Sum({made}) / sum({attempted}); undefined with zero attempts or incomplete paired shooting data", f"{made}, {attempted}", "proportion", "[cutoff - 30 days, cutoff), within block")
    add("pts_mean_7d_minus_30d", "performance", "Recent seven-day mean points minus thirty-day mean points; descriptive within-player contrast", "PTS", "points/appearance", "difference of strictly historical 7-day and 30-day windows")
    add("prior_episode_count", "injury_history", "Count of earlier confirmed injury-related documented episodes since the 2009 audit start; not lifetime clinical history", "Stage 2 event episodes", "episodes")
    add("any_prior_episode", "injury_history", "Indicator for at least one earlier qualifying documented episode", "Stage 2 event episodes", "0/1")
    add("days_since_prior_episode", "injury_history", "Index date minus latest available qualifying episode date; missing if none documented", "Stage 2 event episodes, index_date", "days")
    for w in INJURY_WINDOWS:
        add(f"prior_episodes_{w}d", "injury_history", "Count of qualifying earlier documented episodes", "Stage 2 event episodes", "episodes", f"[cutoff - {w} days, cutoff)")
    return pd.DataFrame(records)


DICTIONARY = feature_dictionary()
FEATURE_NAMES = DICTIONARY.loc[DICTIONARY.retained, "feature"].tolist()
EXCLUDED_CANDIDATES = DICTIONARY.loc[~DICTIONARY.retained, "feature"].tolist()


@dataclass
class FeatureResult:
    X: pd.DataFrame
    candidates: pd.DataFrame
    provenance: pd.DataFrame


def prepare_appearances(box):
    """Retain exactly the positive-minute source history used for Stage 2."""
    from nba_injury.data.design import HOMONYMS
    box=box.copy()
    if "game_date" not in box:
        box["game_date"]=pd.to_datetime(box.GAME_DATE,format="%b %d, %Y",errors="raise")
    box=box.loc[(box.MIN>0)&~box.PLAYER_NAME.isin(HOMONYMS)&box.game_date.ge("2009-01-01")].copy()
    box["player"]=box.PLAYER_NAME
    if box.duplicated(["player","game_date"]).any():
        raise ValueError("Ambiguous player-date source history; do not silently deduplicate.")
    numeric=["MIN",*PERFORMANCE_MEANS.values(),*{c for pair in RATIOS.values() for c in pair}]
    for col in set(numeric):
        box[col]=pd.to_numeric(box[col],errors="raise")
    return box.sort_values(["player","game_date"])


class HistoricalFeatureBuilder:
    """Reusable immutable-source builder with one row-specific cutoff per snapshot.

    Only SNAPSHOT_FIELDS are read from the cohort. Predictive preprocessing is
    separate; there is no learned state, feature selection, or access to labels.
    """
    def __init__(self, appearances, episodes):
        self.games={p:g.sort_values("game_date").copy() for p,g in appearances.groupby("player")}
        events=episodes.loc[episodes.incident_confirmed.eq(True)&episodes.reason.eq("injury_related")].copy()
        events["event_date"]=pd.to_datetime(events.event_date,errors="raise")
        if events.duplicated(["player","event_date"]).any():
            raise ValueError("Qualifying episode dates are not unique.")
        self.episodes={p:g.sort_values("event_date").copy() for p,g in events.groupby("player")}
        self.empty_events=events.iloc[:0]

    def build(self, snapshots):
        missing=set(SNAPSHOT_FIELDS)-set(snapshots.columns)
        if missing:
            raise ValueError(f"Missing required snapshot fields: {sorted(missing)}")
        snapshots=snapshots.loc[:,SNAPSHOT_FIELDS].copy()
        for col in ["index_date","history_cutoff_exclusive","phase_start"]:
            snapshots[col]=pd.to_datetime(snapshots[col],errors="raise")
        if snapshots.observation_id.duplicated().any():
            raise ValueError("Duplicate observation IDs")
        rows,audits=[],[]
        for row in snapshots.itertuples(index=False):
            cutoff=row.history_cutoff_exclusive
            if not row.phase_start<cutoff<row.index_date:
                raise ValueError("Invalid historical cutoff ordering")
            source=self.games.get(row.player)
            if source is None:
                raise ValueError("No appearance history for snapshot")
            # Source filtering happens before any aggregation.
            block=source.loc[source.game_date.ge(row.phase_start)&source.game_date.lt(cutoff)]
            events=self.episodes.get(row.player,self.empty_events)
            past=events.loc[events.event_date.lt(cutoff)]
            if block.empty:
                raise ValueError("Eligible snapshot has no available appearance")
            values={};audit=dict(observation_id=row.observation_id,cutoff=cutoff,
                block_min_source_date=block.game_date.min(),block_max_source_date=block.game_date.max(),
                injury_all_max_source_date=past.event_date.max(),injury_source_rows=len(past))
            windows={}
            for w in WINDOWS:
                lower=cutoff-pd.Timedelta(days=w)
                frame=block.loc[block.game_date.ge(lower)]
                windows[w]=frame
                values[f"appearances_{w}d"]=len(frame)
                values[f"minutes_sum_{w}d"]=float(frame.MIN.sum(min_count=1)) if len(frame) else 0.0
                values[f"minutes_mean_{w}d"]=float(frame.MIN.mean())
                values[f"back_to_back_pairs_{w}d"]=int(frame.game_date.diff().dt.days.eq(1).sum())
                audit[f"workload_{w}d_min_source_date"]=frame.game_date.min()
                audit[f"workload_{w}d_max_source_date"]=frame.game_date.max()
                if len(frame) and not (frame.game_date.ge(lower)&frame.game_date.lt(cutoff)).all():
                    raise AssertionError("Window source date outside its allowed interval")
            values["block_appearances"]=len(block)
            values["block_minutes"]=float(block.MIN.sum(min_count=1))
            values["days_since_available_appearance"]=(row.index_date-block.game_date.max()).days
            perf=windows[30]
            audit["performance_max_source_date"]=perf.game_date.max()
            audit["performance_source_rows_with_missing_values"]=int(perf[list(PERFORMANCE_MEANS.values())].isna().any(axis=1).sum())
            for name,col in PERFORMANCE_MEANS.items():
                values[f"{name}_mean_30d"]=float(perf[col].mean())
            for name,(made,attempted) in RATIOS.items():
                pair=perf[[made,attempted]]
                denominator=pair[attempted].sum(min_count=1)
                values[f"{name}_pct_30d"]=float(pair[made].sum()/denominator) if len(pair) and pair.notna().all().all() and denominator>0 else np.nan
                audit[f"{name}_zero_attempts_30d"]=bool(pd.notna(denominator) and denominator==0)
                audit[f"{name}_missing_source_pairs_30d"]=int(pair.isna().any(axis=1).sum())
            values["pts_mean_7d_minus_30d"]=float(windows[7].PTS.mean()-perf.PTS.mean())
            values["prior_episode_count"]=len(past)
            values["any_prior_episode"]=int(len(past)>0)
            values["days_since_prior_episode"]=(row.index_date-past.event_date.max()).days if len(past) else np.nan
            for w in INJURY_WINDOWS:
                window=past.loc[past.event_date.ge(cutoff-pd.Timedelta(days=w))]
                values[f"prior_episodes_{w}d"]=len(window)
                audit[f"injury_{w}d_max_source_date"]=window.event_date.max()
            max_dates=[v for k,v in audit.items() if k.endswith("max_source_date") and pd.notna(v)]
            if not all(date<cutoff for date in max_dates):
                raise AssertionError("A feature family used current/future source dates")
            rows.append(values);audits.append(audit)
        candidates=pd.DataFrame(rows,index=snapshots.observation_id)
        X=candidates.loc[:,FEATURE_NAMES].copy()
        if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
            raise TypeError("Predictors must be numeric")
        return FeatureResult(X,candidates,pd.DataFrame(audits).set_index("observation_id"))
