"""Build Stage 3 matrices, date audits, and training-only descriptive EDA."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",tempfile.mkdtemp(prefix="nba-stage3-mpl-"))
os.environ.setdefault("XDG_CACHE_HOME",tempfile.mkdtemp(prefix="nba-stage3-cache-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nba_injury.features.historical import (HistoricalFeatureBuilder,prepare_appearances,
    DICTIONARY,FEATURE_NAMES,EXCLUDED_CANDIDATES)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data/processed/stage3")
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    tables=ROOT/"results/tables";figures=ROOT/"results/figures/stage3";figures.mkdir(parents=True,exist_ok=True)
    frozen=[ROOT/"docs/stage2_design_config.json",ROOT/"src/nba_injury/data/design.py",ROOT/"scripts/build_stage2_design.py",ROOT/"results/tables/stage2_design_summary.json"]
    frozen.extend((ROOT/"data/processed/stage2").glob("*.csv"))
    before={str(p.relative_to(ROOT)):digest(p) for p in frozen}
    cohort=pd.read_csv(ROOT/"data/processed/stage2/observations_14d.csv",dtype={"anchor_game_id":str},parse_dates=["index_date","history_cutoff_exclusive","phase_start","anchor_game_date"])
    expected={"train":(4957,593),"validation":(2143,293),"test":(2704,445)}
    for split,(n,positives) in expected.items():
        subset=cohort.loc[cohort.split.eq(split)]
        if (len(subset),int(subset.target_14d.sum()))!=(n,positives):
            raise RuntimeError("Frozen Stage 2 cohort differs; stop and investigate before Stage 3")
    if not ((cohort.index_date-cohort.history_cutoff_exclusive).dt.days.eq(1)&(cohort.index_date-cohort.anchor_game_date).dt.days.eq(2)).all():
        raise RuntimeError("Frozen timing rules differ; stop before changing cohort")
    print("Building fixed historical features; Stage 2 files are read-only...",flush=True)
    box=pd.read_csv(ROOT/"data/raw/nba_player_box_scores.csv",dtype={"Game_ID":str},low_memory=False)
    appearances=prepare_appearances(box)
    episodes=pd.read_csv(ROOT/"data/processed/stage2/event_episodes.csv",parse_dates=["event_date"])
    result=HistoricalFeatureBuilder(appearances,episodes).build(cohort)
    X=result.X;train_mask=cohort.split.eq("train").to_numpy();train=X.iloc[np.flatnonzero(train_mask)]
    y_train=cohort.loc[train_mask,"target_14d"].reset_index(drop=True)
    if not result.candidates.days_since_available_appearance.eq(2).all():
        raise RuntimeError("Unexpected anchoring; investigate before excluding constant candidate")
    if not np.array_equal(X.block_appearances.to_numpy(),cohort.prior_appearances_in_phase.to_numpy()):
        raise RuntimeError("Source-derived appearance history differs from frozen eligibility bookkeeping")
    # Range and nested-window checks do not use outcomes.
    checks={}
    checks['nested_workload_counts']=bool((X.appearances_7d<=X.appearances_14d).all() and (X.appearances_14d<=X.appearances_30d).all() and (X.appearances_30d<=X.block_appearances).all())
    checks['nested_minutes']=bool((X.minutes_sum_7d<=X.minutes_sum_14d).all() and (X.minutes_sum_14d<=X.minutes_sum_30d).all() and (X.minutes_sum_30d<=X.block_minutes).all())
    checks['b2b_pairs_bounded']=all((X[f'back_to_back_pairs_{w}d']<=X[f'appearances_{w}d']-1).all() for w in [7,14,30])
    checks['shooting_proportions_valid']=all(X[c].dropna().between(0,1).all() for c in ['fg_pct_30d','fg3_pct_30d','ft_pct_30d'])
    checks['injury_windows_nested']=all((X[a]<=X[b]).all() for a,b in zip(['prior_episodes_30d','prior_episodes_90d','prior_episodes_180d','prior_episodes_365d'],['prior_episodes_90d','prior_episodes_180d','prior_episodes_365d','prior_episode_count']))
    checks['no_infinite_values']=not np.isinf(X.to_numpy(dtype=float)).any()
    checks['prior_indicator_consistent']=bool((X.any_prior_episode.eq(1)==X.prior_episode_count.gt(0)).all())
    checks['positive_minutes_plausible']=bool(appearances.MIN.between(1,90).all())
    for col in result.provenance:
        if col.endswith('max_source_date'):
            available=result.provenance[col].notna()
            checks[f'{col}_strictly_before_cutoff']=bool((result.provenance.loc[available,col]<result.provenance.loc[available,'cutoff']).all())
    if not all(checks.values()):
        raise AssertionError(f"Sanity/date audit failed: {[k for k,v in checks.items() if not v]}")
    result.provenance.to_csv(out/"source_date_audit.csv")
    result.candidates.loc[:,EXCLUDED_CANDIDATES].to_csv(out/"excluded_candidate_audit.csv")
    dimensions=[];missing=[];prevalence=[]
    for split in expected:
        mask=cohort.split.eq(split).to_numpy();Xs=X.iloc[np.flatnonzero(mask)];metadata=cohort.loc[mask].reset_index(drop=True)
        # IDs/labels/dates are separate, aligned files; X contains only predictors.
        Xs.to_csv(out/f"X_{split}.csv",index=False)
        metadata[['target_14d']].rename(columns={'target_14d':'y'}).to_csv(out/f"y_{split}.csv",index=False)
        metadata.to_csv(out/f"row_manifest_{split}.csv",index=False)
        dimensions.append(dict(split=split,rows=len(Xs),predictors=len(FEATURE_NAMES),positives=int(metadata.target_14d.sum())))
        prevalence.append(dict(split=split,observations=len(metadata),positives=int(metadata.target_14d.sum()),prevalence=float(metadata.target_14d.mean())))
        for col in FEATURE_NAMES:
            missing.append(dict(split=split,feature=col,missing_count=int(Xs[col].isna().sum()),missing_fraction=float(Xs[col].isna().mean())))
    missing=pd.DataFrame(missing);missing.to_csv(tables/'stage3_missingness.csv',index=False)
    pd.DataFrame(prevalence).to_csv(tables/'stage3_prevalence.csv',index=False)
    DICTIONARY.to_csv(tables/'stage3_feature_dictionary.csv',index=False)
    # Historical-era missingness is audited without outcome associations outside train.
    era=[]
    for phase,positions in cohort.groupby('phase_start').indices.items():
        frame=X.iloc[positions]
        for col in FEATURE_NAMES:
            era.append(dict(phase_start=str(phase.date()),feature=col,observations=len(frame),missing_count=int(frame[col].isna().sum()),missing_fraction=float(frame[col].isna().mean())))
    pd.DataFrame(era).to_csv(tables/'stage3_missingness_by_phase.csv',index=False)
    quality=result.provenance.groupby(cohort.split.to_numpy()).agg(rows_with_missing_performance_sources=('performance_source_rows_with_missing_values',lambda s:int(s.gt(0).sum())))
    quality.index.name='split'
    quality.to_csv(tables/'stage3_source_missingness.csv')
    train.describe(percentiles=[.01,.25,.5,.75,.99]).T.to_csv(tables/'stage3_train_distributions.csv')
    sanity=[]
    for col in result.candidates:
        series=result.candidates.iloc[np.flatnonzero(train_mask)][col];freq=series.value_counts(normalize=True,dropna=False)
        sanity.append(dict(feature=col,unique_nonmissing=int(series.nunique()),dominant_value_fraction=float(freq.iloc[0]),near_zero_variance=bool(series.nunique()<=1 or freq.iloc[0]>.995),retained=col in FEATURE_NAMES))
    pd.DataFrame(sanity).to_csv(tables/'stage3_variance_audit.csv',index=False)
    corr=train.corr(method='spearman');pairs=[]
    for i,a in enumerate(FEATURE_NAMES):
        for b in FEATURE_NAMES[i+1:]:
            if pd.notna(corr.loc[a,b]) and abs(corr.loc[a,b])>=.90:
                pairs.append(dict(feature_a=a,feature_b=b,spearman=float(corr.loc[a,b])))
    pd.DataFrame(pairs,columns=['feature_a','feature_b','spearman']).to_csv(tables/'stage3_high_correlations.csv',index=False)
    selected=['appearances_7d','appearances_30d','minutes_sum_7d','minutes_sum_30d','minutes_mean_30d','block_appearances','back_to_back_pairs_30d','pts_mean_30d','pts_mean_7d_minus_30d','fg_pct_30d','prior_episode_count','prior_episodes_365d','days_since_prior_episode']
    training=train.reset_index(drop=True).assign(recorded_event_14d=y_train)
    training.groupby('recorded_event_14d')[selected].agg(['count','mean','median']).to_csv(tables/'stage3_train_by_outcome.csv')
    sample_positions=np.random.default_rng(42).choice(len(train),size=5,replace=False)
    sample=train.iloc[sample_positions][['appearances_7d','minutes_sum_7d','minutes_mean_30d','fg_pct_30d','prior_episode_count','days_since_prior_episode']].reset_index(drop=True)
    sample.insert(0,'example',[f'Example {i+1}' for i in range(5)]);sample.to_csv(tables/'stage3_example_rows.csv',index=False)
    sns.set_theme(style='whitegrid',context='notebook');plt.rcParams.update({'figure.dpi':140,'savefig.dpi':160})
    figure_files=[]
    def save(name):
        plt.tight_layout();path=figures/name;plt.savefig(path,bbox_inches='tight');plt.close();figure_files.append(str(path.relative_to(ROOT)))
    plt.figure(figsize=(7,4));ax=sns.barplot(data=pd.DataFrame(prevalence),x='split',y='prevalence',color='#3b78a4')
    for patch,value in zip(ax.patches,prevalence):ax.annotate(f"{value['prevalence']:.1%}",(patch.get_x()+patch.get_width()/2,patch.get_height()),ha='center',va='bottom')
    ax.set_ylim(0,.20);ax.set_ylabel('Recorded-event prevalence');ax.set_xlabel('');ax.set_title('14-day recorded-event prevalence by chronological split');save('01_prevalence.png')
    def distribution_grid(columns,title,filename):
        fig,axes=plt.subplots(1,len(columns),figsize=(4.1*len(columns),3.7))
        for ax,col in zip(np.atleast_1d(axes),columns):
            sns.boxplot(data=training,x='recorded_event_14d',y=col,showfliers=False,color='#72a7be',ax=ax)
            ax.set_xlabel('Future recorded event (training only)');ax.set_ylabel(col);ax.set_xticks([0,1],['No','Yes'])
        fig.suptitle(title);save(filename)
    distribution_grid(['minutes_sum_7d','minutes_mean_30d','appearances_30d'],'Workload proxies in training observations','02_train_workload.png')
    distribution_grid(['prior_episode_count','prior_episodes_365d','days_since_prior_episode'],'Documented episode history in training observations','03_train_injury_history.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,col in zip(axes,['pts_mean_30d','fg_pct_30d','pts_mean_7d_minus_30d']):
        sns.histplot(data=train,x=col,bins=30,ax=ax,color='#3b78a4');ax.set_ylabel('Training observations')
    fig.suptitle('Selected historical performance distributions (training only)');save('04_train_performance.png')
    missing_pivot=missing.pivot(index='feature',columns='split',values='missing_fraction').reindex(columns=['train','validation','test'])
    nonzero=missing_pivot.loc[missing_pivot.max(axis=1)>0]
    plt.figure(figsize=(7,max(3,nonzero.shape[0]*.8)));sns.heatmap(nonzero*100,annot=True,fmt='.1f',cmap='Blues',cbar_kws={'label':'Missing (%)'});plt.title('Features with missing values');plt.ylabel('');save('05_missingness.png')
    plt.figure(figsize=(11,9));sns.heatmap(corr.loc[selected,selected],vmin=-1,vmax=1,center=0,cmap='vlag',square=True);plt.title('Training-only Spearman correlations');save('06_train_correlations.png')
    after={str(p.relative_to(ROOT)):digest(p) for p in frozen}
    if before!=after:raise AssertionError('Frozen Stage 2 files were changed')
    groups=DICTIONARY.loc[DICTIONARY.retained].groupby('group').size().to_dict()
    summary=dict(stage=3,primary_horizon_days=14,feature_count=len(FEATURE_NAMES),candidate_feature_count=len(DICTIONARY),feature_groups={k:int(v) for k,v in groups.items()},dimensions=dimensions,
        excluded_candidates=EXCLUDED_CANDIDATES,missingness=missing.loc[missing.missing_count>0].to_dict('records'),sanity_checks={k:bool(v) for k,v in checks.items()},
        high_correlation_pairs=len(pairs),figures=figure_files,frozen_stage2_files_unchanged=True,stage2_input_hashes=before,
        models_fitted=False,real_data_preprocessors_fitted=False,feature_selection_used_test_outcomes=False,
        environment=dict(python=sys.version.split()[0],pandas=pd.__version__,numpy=np.__version__,matplotlib=matplotlib.__version__))
    (tables/'stage3_feature_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['feature_count','candidate_feature_count','feature_groups','dimensions','high_correlation_pairs']},indent=2),flush=True)
    print('Missingness:',missing.loc[missing.missing_count>0].to_string(index=False),flush=True)
    print('Frozen cohort unchanged. No model or real-data preprocessor was fitted.',flush=True)


if __name__=='__main__':main()
