"""Checks for the Stage 2 timing/identity/episode rules, without fitting models."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from nba_injury.data.design import IdentityResolver, annotate_episodes, classify_note, label_window, chronological_split


class Stage2DesignTests(unittest.TestCase):
    def test_unmatched_and_homonyms_are_not_assigned(self):
        r=IdentityResolver(["Mike James","Kenyon Martin","Kenyon Martin Jr.","Devyn Marble","D.J. Augustin","Nikola Jokic"])
        self.assertEqual(r.resolve("Mike James (Lamont)")[1],"homonym_excluded")
        self.assertEqual(r.resolve("left wrist injury (DTD)")[1],"unmatched")
        self.assertEqual(r.resolve("Roy Marble")[1],"unmatched")
        self.assertEqual(r.resolve("Kenyon Martin Sr.")[0],"Kenyon Martin")
        self.assertEqual(r.resolve("Roy Devyn Marble / Roy Marble (Devyn)")[0],"Devyn Marble")
        self.assertEqual(r.resolve("D.J. Augustine")[0],"D.J. Augustin")
        self.assertEqual(r.resolve(" Nikola Jokić ")[0],"Nikola Jokic")

    def test_note_semantics(self):
        self.assertEqual(classify_note("activated from IL"),("activation","return"))
        self.assertEqual(classify_note('"activated"'),("activation","return"))
        self.assertEqual(classify_note("placed on IL with sore throat")[1],"illness_or_nonmusculoskeletal")
        self.assertEqual(classify_note("placed on IL with NBA health and safety protocols")[1],"protocol")
        self.assertEqual(classify_note("placed on IL for rest")[1],"rest")
        self.assertEqual(classify_note("placed on IL recovering from surgery on left knee")[1],"continuation_or_recovery")
        self.assertEqual(classify_note("placed on IL")[1],"ambiguous_reason")
        self.assertEqual(classify_note("sprained ankle (DTD)")[0],"uncertain_status")
        self.assertEqual(classify_note("placed on IL with sprained ankle (date approximate)")[0],"uncertain_status")
        self.assertEqual(classify_note("placed on IL with sprained ankle")[1],"injury_related")
        self.assertEqual(classify_note("placed on IL with illness / right knee injury")[1],"ambiguous_reason")

    def test_repeated_placements_and_same_day_appearance(self):
        events=pd.DataFrame([
            dict(player="A",event_date=pd.Timestamp("2021-01-01"),action="placement",reason="injury_related",source_row_id=1,Notes="injury"),
            dict(player="A",event_date=pd.Timestamp("2021-01-03"),action="placement",reason="injury_related",source_row_id=2,Notes="injury again"),
            dict(player="A",event_date=pd.Timestamp("2021-01-05"),action="activation",reason="return",source_row_id=3,Notes="activated"),
            dict(player="A",event_date=pd.Timestamp("2021-01-06"),action="placement",reason="injury_related",source_row_id=4,Notes="injury"),
        ])
        games=pd.DataFrame(dict(player=["A","A"],game_date=pd.to_datetime(["2021-01-01","2021-01-04"])))
        episodes=annotate_episodes(events,games)
        self.assertEqual(episodes.incident_confirmed.tolist(),[False,False,True])
        # Same-day appearance cannot establish active state before an entry.
        self.assertEqual(episodes.iloc[0].prior_state,"unknown")

    def test_same_day_return_and_entry_are_unordered(self):
        date=pd.Timestamp("2021-01-02")
        events=pd.DataFrame([
            dict(player="A",event_date=date,action="placement",reason="injury_related",source_row_id=1,Notes="entry"),
            dict(player="A",event_date=date,action="activation",reason="return",source_row_id=2,Notes="return"),
        ])
        games=pd.DataFrame(dict(player=["A"],game_date=pd.to_datetime(["2021-01-01"])))
        episode=annotate_episodes(events,games).iloc[0]
        self.assertFalse(episode.incident_confirmed)
        self.assertEqual(episode.reason,"same_day_conflicting_events")

    def test_horizons_censor_unknowns_and_terminal_followup(self):
        index=pd.Timestamp("2021-01-01")
        episodes=pd.DataFrame([dict(event_date=pd.Timestamp("2021-01-08"),reason="injury_related",incident_confirmed=True,prior_state="active")])
        end=pd.Timestamp("2021-03-01")
        self.assertEqual(label_window(index,7,episodes,[],[pd.Timestamp("2021-01-07")],end)[0],0)
        self.assertEqual(label_window(index,14,episodes,[],[],end)[0],1)
        self.assertIsNone(label_window(index,7,episodes,[],[],end)[0])
        self.assertIsNone(label_window(index,14,episodes,[pd.Timestamp("2021-01-05")],[],end)[0])
        self.assertIsNone(label_window(index,14,episodes,[pd.Timestamp("2021-01-12")],[],end)[0])
        self.assertIsNone(label_window(index,14,episodes,[],[],pd.Timestamp("2021-01-12"))[0])

    def test_a_known_repeat_is_not_an_incident_or_unknown_target(self):
        episodes=pd.DataFrame([
            dict(event_date=pd.Timestamp("2021-01-03"),reason="injury_related",incident_confirmed=True,prior_state="active"),
            dict(event_date=pd.Timestamp("2021-01-04"),reason="injury_related",incident_confirmed=False,prior_state="inactive"),
            dict(event_date=pd.Timestamp("2021-01-05"),reason="ambiguous_reason",incident_confirmed=False,prior_state="inactive"),
        ])
        label=label_window(pd.Timestamp("2021-01-01"),14,episodes,[],[],pd.Timestamp("2021-02-01"))
        self.assertEqual(label[0],1)
        self.assertEqual(label[2],pd.Timestamp("2021-01-03"))

    def test_chronological_boundaries(self):
        self.assertEqual(chronological_split(pd.Timestamp("2020-02-01")),"train")
        self.assertEqual(chronological_split(pd.Timestamp("2020-12-22")),"validation")
        self.assertEqual(chronological_split(pd.Timestamp("2021-10-19")),"test")
        self.assertEqual(chronological_split(pd.Timestamp("2022-10-18")),"excluded_partial_latest_season")

    def test_future_data_do_not_change_earlier_episode_history(self):
        event=dict(player="A",event_date=pd.Timestamp("2021-01-02"),action="placement",reason="injury_related",source_row_id=1,Notes="entry")
        events=pd.DataFrame([event])
        games=pd.DataFrame(dict(player=["A"],game_date=pd.to_datetime(["2021-01-01"])))
        original=annotate_episodes(events,games)
        future=dict(player="A",event_date=pd.Timestamp("2021-03-01"),action="activation",reason="return",source_row_id=2,Notes="return")
        extended=annotate_episodes(pd.DataFrame([event,future]),pd.concat([games,pd.DataFrame(dict(player=["A"],game_date=pd.to_datetime(["2021-04-01"])))],ignore_index=True))
        pd.testing.assert_frame_equal(original,extended)


if __name__=="__main__":
    unittest.main()
