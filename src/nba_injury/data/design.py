"""Stage 2 design helpers. No feature pipeline or model training is performed."""

import re
import unicodedata
from collections import defaultdict

import pandas as pd


HOMONYMS = {"Chris Johnson", "Chris Wright", "Mike James", "Tony Mitchell"}
MANUAL_ALIASES = {
    "djaugustine": ("D.J. Augustin", "Log spelling differs by a final e; team/date histories checked"),
    "roydevynmarble": ("Devyn Marble", "Log explicitly includes Devyn; Magic/date histories checked"),
    "kenyonmartinsr": ("Kenyon Martin", "Explicit Sr identifies the older player, distinct from Kenyon Martin Jr."),
}
TEAM_CODES = {
    "76ers": {"PHI"}, "Blazers": {"POR"}, "Bobcats": {"CHA"}, "Bucks": {"MIL"},
    "Bullets": {"WAS"}, "Bulls": {"CHI"}, "Cavaliers": {"CLE"}, "Celtics": {"BOS"},
    "Clippers": {"LAC"}, "Grizzlies": {"MEM"}, "Hawks": {"ATL"}, "Heat": {"MIA"},
    "Hornets": {"CHA", "NOH", "NOP", "NOK"}, "Jazz": {"UTA"}, "Kings": {"SAC"},
    "Knicks": {"NYK"}, "Lakers": {"LAL"}, "Magic": {"ORL"}, "Mavericks": {"DAL"},
    "Nets": {"NJN", "BKN", "BRK"}, "Nuggets": {"DEN"}, "Pacers": {"IND"},
    "Pelicans": {"NOP", "NOH"}, "Pistons": {"DET"}, "Raptors": {"TOR"},
    "Rockets": {"HOU"}, "Spurs": {"SAS"}, "Suns": {"PHX", "PHO"},
    "Thunder": {"OKC"}, "Timberwolves": {"MIN"}, "Warriors": {"GSW"}, "Wizards": {"WAS"},
}


def normalize_name(value):
    text = "".join(c for c in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def without_suffix(value):
    return normalize_name(re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", str(value), flags=re.I))


class IdentityResolver:
    """Deterministic identity lookup. Never fuzzy-match or merge homonymous players."""

    def __init__(self, names):
        self.exact = defaultdict(set)
        self.base = defaultdict(set)
        self.names = set(names)
        for name in self.names:
            self.exact[normalize_name(name)].add(name)
            self.base[without_suffix(name)].add(name)

    def resolve(self, raw):
        if pd.isna(raw) or not str(raw).strip():
            return None, "missing", "none"
        parts = [re.sub(r"\([^)]*\)", "", x).strip() for x in str(raw).split("/")]
        candidates, methods = set(), set()
        for part in parts:
            key = normalize_name(part)
            if key in MANUAL_ALIASES:
                name, _ = MANUAL_ALIASES[key]
                if name in self.names:
                    candidates.add(name)
                    methods.add("reviewed_alias")
            elif self.exact[key]:
                candidates.update(self.exact[key])
                methods.add("normalized_exact_or_slash_alias")
        if not candidates:
            candidates = set().union(*(self.base[without_suffix(p)] for p in parts))
            if candidates:
                methods.add("unique_suffix_fallback")
        if len(candidates) != 1:
            return None, "unmatched" if not candidates else "ambiguous", "+".join(sorted(methods)) or "none"
        name = next(iter(candidates))
        if name in HOMONYMS:
            return None, "homonym_excluded", "+".join(sorted(methods))
        return name, "matched", "+".join(sorted(methods))


def classify_note(note):
    """Explicit action and reason rules. Entry date never implies clinical onset."""
    text = str(note).casefold()
    if re.search(r"date approximate|date \d{1,2}-\d{1,2} per", text):
        return "uncertain_status", "no_explicit_placement_action"
    if re.search(r"\bactivated\b|returned to lineup", text):
        return "activation", "return"
    if not re.search(r"placed on (?:il|ir|inactive list|injured list)\b", text):
        return "uncertain_status", "no_explicit_placement_action"
    mixed_injury = bool(re.search(r"injur|sprain|strain|fractur|broken|bruise|contusion|concussion|torn|tear\b|tendinit|tendonit|spasm|sore (?:left |right )?(?:ankle|knee|foot|back|hip|shoulder)",text))
    if re.search(r"health and safety|health/safety|covid|coronavirus|protocol|quarantine", text):
        return "placement", "ambiguous_reason" if mixed_injury else "protocol"
    if re.search(r"\brest\b|\bresting\b|load management", text):
        return "placement", "ambiguous_reason" if mixed_injury else "rest"
    if re.search(r"illness|\bflu\b|virus|infection|fever|pneumonia|respiratory|gastro|strep|food poisoning|migraine|headache|sinus|cold\b|vertigo|kidney|appendic|bronchitis|dehydrat|dehydraft|allerg|asthma|dental|tooth|wisdom teeth|root canal|heart|blood clot|embol|cancer|depression|anxiety|throat|tonsil|stomach|nausea|conjunctivitis|pink eye|mononucleosis|chicken pox|aorta|collapsed lung", text):
        return "placement", "ambiguous_reason" if mixed_injury else "illness_or_nonmusculoskeletal"
    if re.search(r"personal|family|funeral|death|birth|suspend|suspens|disciplin|not with team|leave of absence|military|army|national guard|visa|immigration|legal|conditioning|ineligib|arrest|waived", text):
        return "placement", "other_noninjury"
    if re.search(r"recovering|recovery|rehab|already on|still on|remained on|more weeks|surgery|arthroscop|procedure", text):
        return "placement", "continuation_or_recovery"
    if re.search(r"injur|sprain|strain|fractur|broken|bruise|bone bruse|contusion|concussion|torn|tear\b|tendinitis|tendonitis|tendinopathy|spasm|\bsore\b|soreness|pain|tightness|swelling|swollen|inflammation|dislocat|ruptur|plantar fasciitis|plantar fasciaitis|stress reaction|bone spur|herniat|whiplash|hyperextend|rotator cuff|detached retina|lacerat|abrasion|stitches|hip pointer|turf toe|sports hernia|separated .+shoulder|sublux|shin splint|bursitis|impingement|stiff|bulging disc|effusion|arthritis|\bpulled\b", text):
        return "placement", "injury_related"
    return "placement", "ambiguous_reason"


def assign_phases(games):
    """Identify observed schedule blocks; avoid crossing offseasons/2020 interruption."""
    dates = games["game_date"].drop_duplicates().sort_values()
    starts = dates.loc[dates.diff().dt.days.gt(45) | dates.diff().isna()].tolist()
    ends = dates.shift().loc[dates.diff().dt.days.gt(45)].tolist() + [dates.iloc[-1]]
    records = []
    for n, (start, end) in enumerate(zip(starts, ends)):
        frame = games.loc[games.game_date.between(start, end)]
        records.append(dict(phase_id=n, phase_start=start, phase_end=end,
                            games=frame.Game_ID.nunique(), rows=len(frame),
                            unique_names=frame.PLAYER_NAME.nunique()))
    phases = pd.DataFrame(records)
    games = games.copy()
    games["phase_id"] = -1
    for row in phases.itertuples():
        games.loc[games.game_date.between(row.phase_start, row.phase_end), "phase_id"] = row.phase_id
    return games, phases


def annotate_episodes(status_events, appearances):
    """An incident placement requires a prior observed active state, strictly earlier.

    All placements interrupt active status, regardless of reason. An activation or
    later positive-minute appearance can restore it. Same-day entry/return cannot
    be ordered, and same-day appearances never establish pre-entry active status.
    """
    records = []
    appearance_groups = {k: set(v.game_date) for k, v in appearances.groupby("player")}
    for player, events in status_events.loc[status_events.player.notna()].groupby("player"):
        event_groups = {d: v for d, v in events.groupby("event_date")}
        appearance_dates = appearance_groups.get(player, set())
        state = "unknown"
        for date in sorted(set(event_groups) | appearance_dates):
            frame = event_groups.get(date)
            if frame is not None:
                outs = frame.loc[frame.action.ne("activation")]
                returns = frame.loc[frame.action.eq("activation")]
                if len(outs):
                    reasons = set(outs.reason)
                    mixed = bool(len(returns)) or len(reasons) > 1
                    reason = "same_day_conflicting_events" if mixed else next(iter(reasons))
                    incident = state == "active" and not mixed and set(outs.action) == {"placement"}
                    records.append(dict(player=player, event_date=date, reason=reason,
                                        incident_confirmed=incident, prior_state=state,
                                        source_rows=len(outs), source_ids=";".join(map(str, outs.source_row_id)),
                                        notes=" | ".join(outs.Notes.astype(str))))
                    state = "unknown" if mixed or set(outs.action) != {"placement"} else "inactive"
                elif len(returns):
                    state = "active"
            # A placement on this date has unknown ordering relative to a game.
            if date in appearance_dates and (frame is None or frame.action.eq("activation").all()):
                state = "active"
    return pd.DataFrame(records)


def chronological_split(index_date, train_validation_boundary="2020-12-22",
                        validation_test_boundary="2021-10-19", partial_latest_boundary="2022-10-18"):
    if index_date < pd.Timestamp(train_validation_boundary):
        return "train"
    if index_date < pd.Timestamp(validation_test_boundary):
        return "validation"
    if index_date < pd.Timestamp(partial_latest_boundary):
        return "test"
    return "excluded_partial_latest_season"


def label_window(index_date, horizon, episodes, unresolved_dates, marker_dates, phase_end):
    """Forward data determine outcomes/censoring only, never eligibility/features.

    A zero needs a follow-up marker on/after the horizon end within seven days,
    inside the same observed schedule block. This is an ascertainment proxy, not
    proof of continuous roster membership. Preserve its selection limitation.
    """
    end = index_date + pd.Timedelta(days=horizon - 1)
    if end > phase_end:
        return None, "administrative_followup_incomplete", pd.NaT, end, pd.NaT
    window = episodes.loc[episodes.event_date.between(index_date, end)]
    targets = window.loc[window.incident_confirmed & window.reason.eq("injury_related")]
    # A known inactive -> inactive update is a repeat, not a new placement and
    # not a reason to discard an already observed incident target. Unknown
    # initial states and unordered same-day transitions remain uncertain.
    unclear = window.loc[
        window.reason.eq("same_day_conflicting_events") |
        (window.prior_state.ne("inactive") & window.reason.isin({
            "ambiguous_reason", "continuation_or_recovery", "no_explicit_placement_action"})) |
        (window.prior_state.eq("unknown") & window.reason.eq("injury_related"))]
    unknown_dates = [d for d in unresolved_dates if index_date <= d <= end]
    unclear_dates = list(unclear.event_date) + unknown_dates
    first_unclear = min(unclear_dates) if unclear_dates else pd.NaT
    first_target = targets.event_date.min() if len(targets) else pd.NaT
    # Apply source-quality exclusions to both classes, even when a known target
    # precedes an unknown event. This avoids selectively retaining those cases.
    if pd.notna(first_unclear):
        return None, "ambiguous_or_unresolved_forward_event", first_unclear, end, pd.NaT
    if pd.notna(first_target):
        return 1, "observed_injury_related_episode", first_target, end, pd.NaT
    markers = [d for d in marker_dates if end <= d <= min(end + pd.Timedelta(days=7), phase_end)]
    if not markers:
        return None, "no_nearby_forward_observation", pd.NaT, end, pd.NaT
    return 0, "no_documented_target_with_observed_followup", pd.NaT, end, min(markers)
