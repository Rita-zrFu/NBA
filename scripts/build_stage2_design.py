"""Audit event semantics/identity and build candidate observations and temporal splits.

No predictors, preprocessing transformations, or models are fitted here.
"""

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nba_injury.data.design import (HOMONYMS, MANUAL_ALIASES, TEAM_CODES,
    IdentityResolver, annotate_episodes, assign_phases, chronological_split,
    classify_note, label_window, normalize_name)


def summarize(frame, horizon):
    col = f"target_{horizon}d"
    valid = frame.loc[frame[col].notna()]
    n, pos = len(valid), int(valid[col].sum())
    counts = valid.groupby("player").size()
    return dict(horizon_days=horizon, observations=n, positive_outcomes=pos,
                negative_outcomes=n-pos, prevalence=pos/n if n else None,
                unique_players=int(valid.player.nunique()),
                first_index_date=str(valid.index_date.min().date()) if n else None,
                last_index_date=str(valid.index_date.max().date()) if n else None,
                observations_per_player={"min":int(counts.min()), "median":float(counts.median()),
                    "mean":float(counts.mean()), "p90":float(counts.quantile(.9)), "max":int(counts.max())} if n else {},
                excluded_or_censored=frame[f"label_status_{horizon}d"].value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/stage2")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tables = ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "docs/stage2_design_config.json").read_text())
    horizons = config["horizons_days"]
    if horizons != [7,14] or config["snapshot_spacing_days"] < max(horizons):
        raise ValueError("This audited design preserves 7/14-day horizons with nonoverlapping windows.")
    if config["index_offset_from_anchor_days"] != 2 or config["feature_availability_lag_days"] != 1:
        raise ValueError("This audited date-only design uses a one-day availability buffer.")
    if config["negative_followup_marker_grace_days"] != 7:
        raise ValueError("The audited label rule uses seven days of marker grace.")
    print("Reading immutable sources and auditing identities...", flush=True)
    raw = ROOT / "data/raw"
    box = pd.read_csv(raw / "nba_player_box_scores.csv", dtype={"Game_ID":str}, low_memory=False)
    box["game_date"] = pd.to_datetime(box.GAME_DATE, format="%b %d, %Y", errors="raise")
    box = box.loc[box.game_date.ge(config["source_start"])].copy()
    resolver = IdentityResolver(box.PLAYER_NAME.unique())
    # Audit the independently retained identifiers that expose homonymous names.
    name_ids = defaultdict(set)
    with zipfile.ZipFile(raw / "supplementary_season_stats_2009_2023.zip") as archive:
        for filename in archive.namelist():
            frame = pd.read_csv(archive.open(filename))
            for name, identifier in zip(frame.Player,frame["Player-additional"]):
                name_ids[normalize_name(name)].add(identifier)
    homonym_evidence = [{"box_name":name,"distinct_ids":len(name_ids[normalize_name(name)]),
                        "identifiers":";".join(sorted(name_ids[normalize_name(name)]))} for name in sorted(HOMONYMS)]
    assert all(row["distinct_ids"] > 1 for row in homonym_evidence)
    pd.DataFrame(homonym_evidence).to_csv(tables / "stage2_homonym_evidence.csv",index=False)
    box, phases = assign_phases(box)
    phases.to_csv(tables / "stage2_schedule_phases.csv", index=False)
    logs = pd.read_csv(raw / "nba_injury_logs.csv")
    logs["source_row_id"] = np.arange(len(logs))
    log_cols = [c for c in logs if not c.startswith("Unnamed") and c != "source_row_id"]
    duplicate_count = int(logs.duplicated(log_cols).sum())
    logs = logs.drop_duplicates(log_cols).copy()
    logs["event_date"] = pd.to_datetime(logs.Date, errors="raise")
    logs = logs.loc[logs.event_date.ge(config["source_start"])].copy()
    classified = logs.Notes.map(classify_note)
    logs[["action", "reason"]] = pd.DataFrame(classified.tolist(), index=logs.index)
    status_records, identity_records = [], []
    for row in logs.itertuples():
        # Notes determine action even for the documented direction anomaly.
        raw_name = row.Acquired if row.action == "activation" and pd.notna(row.Acquired) else row.Relinquished
        if pd.isna(raw_name):
            raw_name = row.Acquired
        player, status, method = resolver.resolve(raw_name)
        action,reason=row.action,row.reason
        if action=="placement" and pd.isna(row.Relinquished):
            action,reason="uncertain_status","no_explicit_placement_action"
        identity_records.append(dict(source_row_id=row.source_row_id, event_date=row.event_date,
            direction="Acquired" if pd.notna(row.Acquired) and raw_name==row.Acquired else "Relinquished",
            raw_name=raw_name, player=player, match_status=status, match_method=method,
            Team=row.Team, action=action, reason=reason, Notes=row.Notes))
        status_records.append(dict(source_row_id=row.source_row_id, event_date=row.event_date,
            player=player, match_status=status, Team=row.Team, action=action, reason=reason, Notes=row.Notes))
    identities = pd.DataFrame(identity_records)
    statuses = pd.DataFrame(status_records)
    identities.to_csv(tables / "stage2_identity_audit.csv", index=False)
    identities.loc[identities.match_status.ne("matched")].to_csv(tables / "stage2_unresolved_identities.csv", index=False)
    alias_evidence=[]
    for row in identities.loc[identities.match_method.str.contains("reviewed_alias")].itertuples():
        context=box.loc[box.PLAYER_NAME.eq(row.player)&box.Team.isin(TEAM_CODES.get(row.Team,set()))]
        distances=(context.game_date-row.event_date).dt.days.abs()
        nearest=context.loc[distances.idxmin(),"game_date"] if len(context) else pd.NaT
        alias_evidence.append(dict(raw_name=row.raw_name,canonical=row.player,event_date=row.event_date,
            log_team=row.Team,nearest_same_team_appearance=nearest,
            distance_days=int(distances.min()) if len(context) else None))
    pd.DataFrame(alias_evidence).to_csv(tables / "stage2_reviewed_alias_evidence.csv",index=False)
    notes = logs.groupby(["action", "reason", "Notes"]).agg(rows=("source_row_id","size"), first_date=("event_date","min"), last_date=("event_date","max")).reset_index()
    notes.to_csv(tables / "stage2_note_inventory.csv", index=False)
    notes.groupby(["action","reason"]).rows.sum().reset_index().to_csv(tables / "stage2_note_rule_counts.csv", index=False)
    reporting=[]
    for phase in phases.itertuples():
        entries=logs.loc[logs.event_date.between(phase.phase_start,phase.phase_end)&logs.action.ne("activation")]
        counts=entries.reason.value_counts().to_dict()
        uncertain=entries.reason.isin(["ambiguous_reason","continuation_or_recovery","no_explicit_placement_action"]).sum()
        reporting.append(dict(phase_id=phase.phase_id,phase_start=phase.phase_start,phase_end=phase.phase_end,
            outgoing_rows=len(entries),injury_related_rows=counts.get("injury_related",0),
            uncertain_or_recovery_rows=int(uncertain),uncertain_or_recovery_fraction=float(uncertain/len(entries)) if len(entries) else None,
            **{f"reason_{key}":value for key,value in counts.items()}))
    pd.DataFrame(reporting).to_csv(tables / "stage2_reporting_by_phase.csv",index=False)
    # Box-score appearances are the only anchors; never invent nonappearance days.
    appearances = box.loc[(box.MIN > 0) & ~box.PLAYER_NAME.isin(HOMONYMS)].copy()
    appearances["player"] = appearances.PLAYER_NAME
    if appearances.duplicated(["player","game_date"]).any():
        raise ValueError("Ambiguous player-date anchors remain after homonym exclusions.")
    print("Reconstructing documented active/inactive transitions...", flush=True)
    episodes = annotate_episodes(statuses, appearances)
    episodes.to_csv(out / "event_episodes.csv", index=False)
    statuses.to_csv(out / "resolved_status_events.csv", index=False)
    ep_summary = episodes.groupby(["reason","incident_confirmed","prior_state"]).size().reset_index(name="events")
    ep_summary.to_csv(tables / "stage2_episode_audit.csv", index=False)
    outgoing = statuses.loc[statuses.action.ne("activation")]
    conflicts = set(zip(outgoing.loc[outgoing.player.notna(),"player"], outgoing.loc[outgoing.player.notna(),"event_date"]))
    # Unmatched outbound events can otherwise become false negatives for a player.
    # Quarantine every league-wide window, including possible transfers whose
    # future team cannot be established from a full roster. Never fuzzy-assign.
    unresolved = outgoing.loc[outgoing.player.isna() & outgoing.match_status.ne("homonym_excluded")]
    unknown_dates = sorted(set(unresolved.event_date))
    phase_lookup = phases.set_index("phase_id").to_dict("index")
    episode_lookup = {k: v for k,v in episodes.groupby("player")}
    empty_episodes = episodes.iloc[:0]
    event_markers = statuses.loc[statuses.player.notna() & statuses.action.isin(["activation","placement"])]
    marker_lookup = {k:set(v.event_date) for k,v in event_markers.groupby("player")}
    selected = []
    exclusion_counts = {}
    anchor_total = 0
    # Sampling uses only history and index dates, before computing any labels.
    for (player, phase_id), frame in appearances.groupby(["player","phase_id"]):
        phase = phase_lookup[phase_id]
        if not pd.Timestamp(config["first_candidate_phase_start"]) <= phase["phase_start"] <= pd.Timestamp(config["last_candidate_phase_start"]):
            continue
        frame = frame.sort_values("game_date")
        marker_dates = sorted(set(frame.game_date) | {
            d for d in marker_lookup.get(player,set()) if phase["phase_start"] <= d <= phase["phase_end"]})
        previous_selected_index = pd.NaT
        for count, row in enumerate(frame.itertuples(), start=1):
            anchor_total += 1
            index_date = row.game_date + pd.Timedelta(days=config["index_offset_from_anchor_days"])
            feature_cutoff=index_date-pd.Timedelta(days=config["feature_availability_lag_days"])
            reason = None
            if count < config["minimum_prior_appearances_in_phase"]:
                reason = "insufficient_prior_appearances"
            elif (feature_cutoff - phase["phase_start"]).days < config["minimum_phase_history_days"]:
                reason = "insufficient_phase_history_days"
            elif any((player,d) in conflicts for d in pd.date_range(row.game_date,index_date-pd.Timedelta(days=1))):
                reason = "outgoing_or_uncertain_status_before_index"
            elif any(d in unknown_dates for d in pd.date_range(row.game_date,index_date-pd.Timedelta(days=1))):
                reason = "unresolved_league_event_before_index"
            elif index_date + pd.Timedelta(days=max(horizons)-1) > phase["phase_end"]:
                reason = "maximum_horizon_crosses_phase_end"
            elif pd.notna(previous_selected_index) and (index_date-previous_selected_index).days < config["snapshot_spacing_days"]:
                reason = "spacing_less_than_14_days"
            if reason:
                exclusion_counts[reason] = exclusion_counts.get(reason,0) + 1
                continue
            previous_selected_index = index_date
            rec = dict(observation_id=f"{normalize_name(player)}_{index_date:%Y%m%d}", player=player,
                anchor_game_id=row.Game_ID, anchor_game_date=row.game_date, index_date=index_date,
                history_cutoff_exclusive=feature_cutoff, anchor_team=row.Team,
                phase_id=phase_id, phase_start=phase["phase_start"], phase_end=phase["phase_end"],
                prior_appearances_in_phase=count, split=chronological_split(index_date,
                    config["train_validation_boundary"],config["validation_test_boundary"],config["test_partial_latest_boundary"]))
            for h in horizons:
                target, status, event_date, end, witness_date = label_window(index_date,h,
                    episode_lookup.get(player,empty_episodes), unknown_dates, marker_dates,phase["phase_end"])
                resolution_date=event_date if target==1 else witness_date if target==0 else pd.NaT
                rec.update({f"target_{h}d":target, f"label_status_{h}d":status,
                    f"outcome_date_{h}d":event_date, f"horizon_end_{h}d":end,
                    f"negative_followup_witness_{h}d":witness_date,
                    f"assumed_label_available_date_{h}d":resolution_date+pd.Timedelta(days=2) if pd.notna(resolution_date) else pd.NaT})
            selected.append(rec)
    obs = pd.DataFrame(selected).sort_values(["index_date","player"]).reset_index(drop=True)
    for h in horizons:
        obs[f"target_{h}d"] = obs[f"target_{h}d"].astype("Int64")
    obs.to_csv(out / "candidate_observations.csv", index=False)
    summary = dict(config=config, exact_duplicate_log_rows_removed=duplicate_count,
        identity_audit_period=[str(logs.event_date.min().date()),str(logs.event_date.max().date())],
        identity_coverage=identities.groupby(["direction","match_status"]).size().reset_index(name="rows").to_dict("records"),
        homonym_names_excluded=sorted(HOMONYMS), homonym_box_rows=int(box.PLAYER_NAME.isin(HOMONYMS).sum()),
        reviewed_aliases={k:{"canonical":v[0],"reason":v[1]} for k,v in MANUAL_ALIASES.items()},
        unresolved_outgoing_dates=[str(d.date()) for d in unknown_dates],
        notes_rule_counts=logs.groupby(["action","reason"]).size().reset_index(name="rows").to_dict("records"),
        candidate_appearance_anchors=anchor_total, historical_eligibility_exclusions=exclusion_counts,
        selected_snapshots=len(obs), horizons=[], overlaps={}, integrity={})
    cohort_rows = []
    for h in horizons:
        overall = summarize(obs,h)
        summary["horizons"].append(overall)
        cohort_rows.append(dict(split="all",**{k:v for k,v in overall.items() if k not in {"observations_per_player","excluded_or_censored"}}))
        valid = obs.loc[obs[f"target_{h}d"].notna()]
        valid.to_csv(out / f"observations_{h}d.csv",index=False)
        players = {}
        for split in ["train","validation","test"]:
            split_obs=valid.loc[valid.split.eq(split)]
            split_obs.to_csv(out / f"{split}_{h}d.csv",index=False)
            result=summarize(split_obs,h)
            cohort_rows.append(dict(split=split,**{k:v for k,v in result.items() if k not in {"observations_per_player","excluded_or_censored"}}))
            players[split]=set(split_obs.player)
        summary["overlaps"][str(h)]={
            "train_validation":len(players["train"]&players["validation"]),
            "train_test":len(players["train"]&players["test"]),
            "validation_test":len(players["validation"]&players["test"]),
            "all_three":len(set.intersection(*players.values())),
            "test_players_seen_in_train":len(players["test"]&players["train"]),
            "test_players_not_seen_in_train":len(players["test"]-players["train"])}
    cohort_table=pd.DataFrame(cohort_rows)
    cohort_table.to_csv(tables / "stage2_cohort_summary.csv",index=False)
    obs.loc[obs.target_14d.notna()].groupby(["player","split"]).size().unstack(fill_value=0).to_csv(tables / "stage2_observations_per_player.csv")
    # Mechanical leakage checks; model validation remains a later stage.
    assert not obs.observation_id.duplicated().any()
    assert (obs.anchor_game_date < obs.index_date).all()
    assert (obs.anchor_game_date < obs.history_cutoff_exclusive).all()
    assert (obs.history_cutoff_exclusive < obs.index_date).all()
    assert (obs.horizon_end_14d <= obs.phase_end).all()
    assert obs.groupby("player").index_date.apply(lambda s: s.sort_values().diff().dropna().dt.days.ge(14).all()).all()
    for h in horizons:
        valid=obs.loc[obs[f"target_{h}d"].notna()]
        positives=valid.loc[valid[f"target_{h}d"].eq(1)]
        negatives=valid.loc[valid[f"target_{h}d"].eq(0)]
        assert (positives[f"outcome_date_{h}d"]>=positives.index_date).all()
        assert (positives[f"outcome_date_{h}d"]<=positives[f"horizon_end_{h}d"]).all()
        assert not positives.duplicated(["player",f"outcome_date_{h}d"]).any()
        assert (negatives[f"negative_followup_witness_{h}d"]>=negatives[f"horizon_end_{h}d"]).all()
        assert (negatives[f"negative_followup_witness_{h}d"]<=negatives[f"horizon_end_{h}d"]+pd.Timedelta(days=7)).all()
        for earlier,later in [("train","validation"),("validation","test")]:
            assert valid.loc[valid.split.eq(earlier),f"horizon_end_{h}d"].max() < valid.loc[valid.split.eq(later),"index_date"].min()
            assert valid.loc[valid.split.eq(earlier),f"assumed_label_available_date_{h}d"].max() < valid.loc[valid.split.eq(later),"index_date"].min()
    common=obs.loc[obs.target_7d.notna()&obs.target_14d.notna()]
    assert (common.target_7d <= common.target_14d).all()
    summary["common_labelable_cohort"]={"observations":len(common),"unique_players":int(common.player.nunique()),
        "positive_7d":int(common.target_7d.sum()),"prevalence_7d":float(common.target_7d.mean()),
        "positive_14d":int(common.target_14d.sum()),"prevalence_14d":float(common.target_14d.mean())}
    pd.DataFrame([summary["common_labelable_cohort"]]).to_csv(tables / "stage2_common_horizon_summary.csv",index=False)
    summary["integrity"]={"unique_observation_ids":True,"prior_date_anchors_only":True,"one_day_availability_buffer":True,
        "minimum_spacing_14_days":True,"horizons_within_schedule_phase":True,
        "split_forward_windows_do_not_overlap":True,"training_labels_available_before_next_period":True,
        "negatives_have_recorded_followup_witnesses":True,"nested_horizons_consistent":True,
        "models_trained":False,"full_features_engineered":False}
    (tables / "stage2_design_summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    print(cohort_table.to_string(index=False),flush=True)
    print("Historical eligibility exclusions:",exclusion_counts,flush=True)
    print("Overlap:",summary["overlaps"],flush=True)
    print("Done. No feature pipeline or model training was run.",flush=True)


if __name__ == "__main__":
    main()
