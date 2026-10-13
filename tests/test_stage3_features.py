"""Boundary, prefix-invariance, schema, and training-only preprocessing tests."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from nba_injury.features.historical import (HistoricalFeatureBuilder,FEATURE_NAMES,
    PERFORMANCE_MEANS,RATIOS,prepare_appearances)
from nba_injury.features.preprocessing import make_preprocessor


def game(date,minutes=20,pts=10,made=1,attempts=2):
    row=dict(player="Example",PLAYER_NAME="Example",GAME_DATE=pd.Timestamp(date).strftime("%b %d, %Y").upper(),
        game_date=pd.Timestamp(date),MIN=minutes)
    for col in set(PERFORMANCE_MEANS.values())|{c for pair in RATIOS.values() for c in pair}:
        row[col]=1.0
    row.update(PTS=pts,FGM=made,FGA=attempts,FG3M=0,FG3A=0,FTM=0,FTA=0)
    return row


def episode(date,qualifies=True,reason="injury_related"):
    return dict(player="Example",event_date=pd.Timestamp(date),incident_confirmed=qualifies,reason=reason)


class HistoricalFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.snapshots=pd.DataFrame([dict(observation_id="sample",player="Example",index_date=pd.Timestamp("2020-02-10"),
            history_cutoff_exclusive=pd.Timestamp("2020-02-09"),phase_start=pd.Timestamp("2020-01-01"))])
        self.games=pd.DataFrame([game("2020-02-02",10,5,1,1),game("2020-02-03",20,10,0,9),game("2020-02-08",30,15,1,2)])
        self.episodes=pd.DataFrame([episode("2020-01-10"),episode("2020-02-09"),episode("2020-02-15"),episode("2020-02-07",False),episode("2020-02-06",True,"rest")])

    def test_future_games_and_events_cannot_change_features(self):
        first=HistoricalFeatureBuilder(self.games,self.episodes).build(self.snapshots)
        # Cutoff-day and index/target-day data are excluded, however extreme.
        extra=pd.DataFrame([game("2020-02-09",90,9999),game("2020-02-10",90,9999),game("2021-01-01",90,9999)])
        extra_events=pd.DataFrame([episode("2020-02-10"),episode("2020-02-20")])
        second=HistoricalFeatureBuilder(pd.concat([self.games,extra],ignore_index=True),pd.concat([self.episodes,extra_events],ignore_index=True)).build(self.snapshots)
        pd.testing.assert_frame_equal(first.X,second.X)
        self.assertEqual(first.X.loc["sample","prior_episode_count"],1)
        self.assertEqual(first.X.loc["sample","days_since_prior_episode"],31)
        for col in first.provenance:
            if col.endswith("max_source_date"):
                dates=first.provenance[col].dropna()
                self.assertTrue((dates<pd.Timestamp("2020-02-09")).all())

    def test_windows_weighted_shooting_and_back_to_back(self):
        result=HistoricalFeatureBuilder(self.games,self.episodes).build(self.snapshots)
        x=result.X.iloc[0]
        self.assertEqual(x.appearances_7d,3)  # Lower boundary Feb 2 included.
        self.assertEqual(x.minutes_sum_7d,60)
        self.assertEqual(x.back_to_back_pairs_7d,1)
        self.assertAlmostEqual(x.fg_pct_30d,2/12)
        self.assertTrue(pd.isna(x.fg3_pct_30d))
        self.assertEqual(x.prior_episodes_30d,1)

    def test_no_past_episode_remains_missing_not_zero_days(self):
        future_only=pd.DataFrame([episode("2020-02-09")])
        x=HistoricalFeatureBuilder(self.games,future_only).build(self.snapshots).X.iloc[0]
        self.assertEqual(x.prior_episode_count,0)
        self.assertEqual(x.any_prior_episode,0)
        self.assertTrue(pd.isna(x.days_since_prior_episode))

    def test_metadata_mutations_have_no_effect_and_are_not_predictors(self):
        polluted=self.snapshots.assign(split="test",target_14d=1,outcome_date_14d="2099-01-01",
            phase_end="2099-01-01",negative_followup_witness_14d="2099-01-01",assumed_label_available_date_14d="2099-01-01",prior_appearances_in_phase=99999)
        builder=HistoricalFeatureBuilder(self.games,self.episodes)
        pd.testing.assert_frame_equal(builder.build(self.snapshots).X,builder.build(polluted).X)
        self.assertEqual(list(builder.build(polluted).X),FEATURE_NAMES)
        self.assertNotIn("days_since_available_appearance",FEATURE_NAMES)

    def test_preprocessing_does_not_learn_from_validation_or_test(self):
        train=pd.DataFrame(np.tile([1.,2.,3.],(len(FEATURE_NAMES),1)).T,columns=FEATURE_NAMES)
        validation=pd.DataFrame(10000.,index=range(2),columns=FEATURE_NAMES)
        validation.loc[:,"days_since_prior_episode"]=np.nan
        pipeline=make_preprocessor("linear").fit(train)
        stats=pipeline.named_steps['impute'].statistics_.copy()
        means=pipeline.named_steps['scale'].mean_.copy()
        transformed=pipeline.transform(validation)
        self.assertEqual(transformed.shape,(2,len(FEATURE_NAMES)))
        np.testing.assert_array_equal(stats,pipeline.named_steps['impute'].statistics_)
        np.testing.assert_array_equal(means,pipeline.named_steps['scale'].mean_)
        self.assertEqual(transformed[0,FEATURE_NAMES.index("days_since_prior_episode")],0.)

    def test_preprocessing_rejects_labels_split_and_identity_columns(self):
        X=pd.DataFrame(0.,index=range(2),columns=FEATURE_NAMES)
        for forbidden in ["target_14d","split","player","outcome_date_14d","history_cutoff_exclusive"]:
            with self.assertRaises(ValueError):
                make_preprocessor("tree").fit(X.assign(**{forbidden:1}))

    def test_ambiguous_source_dates_are_not_silently_deduplicated(self):
        with self.assertRaises(ValueError):
            prepare_appearances(pd.concat([self.games,self.games.iloc[:1]],ignore_index=True))

    def test_invalid_cutoff_is_rejected(self):
        bad=self.snapshots.copy();bad["history_cutoff_exclusive"]=bad.index_date
        with self.assertRaises(ValueError):
            HistoricalFeatureBuilder(self.games,self.episodes).build(bad)

    def test_sklearn_cloning_leaves_new_fold_preprocessor_unfitted(self):
        X=pd.DataFrame(1.,index=range(3),columns=FEATURE_NAMES)
        original=make_preprocessor("tree").fit(X)
        fresh=clone(original)
        self.assertFalse(hasattr(fresh.named_steps['impute'],'statistics_'))
        self.assertEqual(list(original.get_feature_names_out()),FEATURE_NAMES)


if __name__=="__main__":
    unittest.main()
