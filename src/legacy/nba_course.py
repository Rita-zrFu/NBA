#!/usr/bin/env python
# coding: utf-8

# In[4]:


def clean_player_name(name):
    return str(name).upper().strip()

def classify_injury_note(note, injury_dict):
    note = str(note).lower()
    for group, keywords in injury_dict.items():
        if any(kw in note for kw in keywords):
            return group
    return None


# In[5]:


# ============================================
# NBA Injury Project
# library：pandas, numpy, matplotlib, seaborn
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.1)

# --------------- 1. read data -----------------
inj_path = '/Users/ritaronaldo/Library/CloudStorage/OneDrive-Emory/COURSE/585/final proj/NBA Player Injury Stats(1951 - 2023).csv'
stats_path = '/Users/ritaronaldo/Library/CloudStorage/OneDrive-Emory/COURSE/585/final proj/NBA Player Stats(1950 - 2022).csv'
box_path = '/Users/ritaronaldo/Library/CloudStorage/OneDrive-Emory/COURSE/585/final proj/NBA Player Box Score Stats(1950 - 2022).csv'

inj = pd.read_csv(inj_path)
stats = pd.read_csv(stats_path)
box = pd.read_csv(box_path)

inj = inj.drop(columns=[col for col in inj.columns if col.startswith("Unnamed")], errors="ignore")
stats = stats.drop(columns=[col for col in stats.columns if col.startswith("Unnamed")], errors="ignore")
box = box.drop(columns=[col for col in box.columns if col.startswith("Unnamed")], errors="ignore")

print("Injury columns:", inj.columns.tolist())
print("Season stats columns:", stats.columns.tolist())
print("Box score columns:", box.columns.tolist())


# In[6]:


# --------------- 2. preprocessing -----------------
# 2.1 date
inj["Date"] = pd.to_datetime(inj["Date"], errors="coerce")
inj["year"] = inj["Date"].dt.year
inj = inj.dropna(subset=["Date"])  # drop date NA

# 2.2 player name formate
inj["player_upper"] = inj["Relinquished"].str.upper().str.strip()
stats["player_upper"] = stats["Player"].str.upper().str.strip()
box["player_upper"] = box["PLAYER_NAME"].str.upper().str.strip()

# 2.3 season year, i.e. "2015-16" -> 2015
def season_to_year(season):
    # season "2015-16"
    try:
        return int(str(season)[:4])
    except:
        return np.nan

stats["season_year"] = stats["Season"].apply(season_to_year)
box["season_year"] = box["Season"].apply(season_to_year)

# only keep year
stats = stats.dropna(subset=["season_year"])
box = box.dropna(subset=["season_year"])
stats["season_year"] = stats["season_year"].astype(int)
box["season_year"] = box["season_year"].astype(int)


# In[7]:


# --------------- 3. from Notes extract injuries type -----------------
# keyword 
inj["Notes_clean"] = inj["Notes"].fillna("").str.lower()

injury_dict = {
    "Ankle/Foot": ["ankle", "foot", "toe", "achilles", "heel", "plantar"],
    "Knee/Leg": ["knee", "leg", "calf", "thigh", "hamstring", "quadricep", "groin", "hip"],
    "Back": ["back", "spine", "disc"],
    "Shoulder/Arm": ["shoulder", "arm", "bicep", "tricep", "elbow", "wrist", "hand", "finger", "thumb"],
    "Head/Neck": ["concussion", "head", "neck", "face", "nose", "eye"],
    "Illness": ["illness", "flu", "virus", "covid", "cold", "infection", "fever", "pneumonia"],
    "Torso/Core": ["rib", "chest", "abdomen", "oblique", "core", "stomach"],
}

def classify_injury(note):
    """
    match keywords, return injury
    if no, return None。
    """
    for group, keywords in injury_dict.items():
        for kw in keywords:
            if kw in note:
                return group
    return None

inj["injury_group"] = inj["Notes_clean"].apply(classify_injury)

# only keep records with specific injury type
inj_injury = inj.dropna(subset=["injury_group"]).copy()

print("Total injury rows with classified type:", len(inj_injury))


# In[8]:


# ======================================================
# EDA 1: distribution of statistics
# ======================================================
# only see modern data
stats_eda = stats[stats["season_year"].between(2002, 2022)].copy()

# choose some meaningful numerical variables
eda_cols = [
    "Age", "G", "GS", "MP", 
    "PTS", "FGA", "TRB", "AST", 
    "STL", "BLK", "TOV", "PF"
]

# ensure these columns exist
eda_cols = [c for c in eda_cols if c in stats_eda.columns]
print("columns：", eda_cols)

n_cols = 4  # each row has 4 plots
n_rows = int(np.ceil(len(eda_cols) / n_cols))

plt.figure(figsize=(4 * n_cols, 3 * n_rows))

for i, col in enumerate(eda_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(stats_eda[col], bins=30, kde=False)
    plt.title(col)
    plt.xlabel("")
    plt.ylabel("Count")

plt.suptitle("Distribution of Season-level Stats (2002–2022)", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()


# In[9]:


# ======================================================
# EDA2: trend of injury records with years
# see if there's abnormal year/ data quality issue
# ======================================================

inj_year_counts = (
    inj_injury
    .groupby("year")
    .size()
    .reset_index(name="injury_records")
)

plt.figure(figsize=(8, 4))
sns.lineplot(data=inj_year_counts, x="year", y="injury_records", marker="o")
plt.title("Number of Classified Injury Records by Year")
plt.xlabel("Year")
plt.ylabel("Number of injury records")
plt.tight_layout()
plt.show()


# In[10]:


# ======================================================
# EDA 3: heatmap between variables
# ======================================================

corr_df = stats_eda[eda_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation between Season-level Features (2002–2022)")
plt.tight_layout()
plt.show()


# In[11]:


# ======================================================
# injury type between different years
# ======================================================

# decade: unit，eg 1980s, 1990s ...
inj_injury = inj_injury.dropna(subset=["year"])
inj_injury["decade"] = (inj_injury["year"] // 10) * 10

# only see records after 1980 (more complete)
inj_injury_dec = inj_injury[inj_injury["year"] >= 1980].copy()

# 1) every decade + injury_group injury records
decade_group_counts = (
    inj_injury_dec
    .groupby(["decade", "injury_group"])
    .size()
    .reset_index(name="inj_count")
)
# 2) how many independent players in every decade
stats_dec = stats.copy()
stats_dec["decade"] = (stats_dec["season_year"] // 10) * 10
stats_dec = stats_dec[stats_dec["decade"] >= 1980]

players_per_decade = (
    stats_dec
    .groupby("decade")["player_upper"]
    .nunique()
    .reset_index(name="n_players")
)
#3) merge: injury numbers every 100 players
decade_group_counts = decade_group_counts.merge(players_per_decade, on="decade", how="left")
decade_group_counts["inj_per_100_players"] = decade_group_counts["inj_count"] / decade_group_counts["n_players"] * 100
# only draw top 5 injuries
top_groups = (
    inj_injury_dec["injury_group"]
    .value_counts()
    .head(5)
    .index
)
decade_group_top = decade_group_counts[decade_group_counts["injury_group"].isin(top_groups)]

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=decade_group_top,
    x="decade",
    y="inj_per_100_players",
    hue="injury_group",
    marker="o"
)
plt.title("NBA Injury Rate per 100 Players by Decade (Top 5 Types)")
plt.xlabel("Decade")
plt.ylabel("Injuries per 100 player-seasons")
plt.xticks(sorted(decade_group_top["decade"].unique()))
plt.legend(title="Injury Type")
plt.tight_layout()
plt.show()


# In[12]:


# ======================================================
# Q2 Alternative: bar charts by group
# ======================================================

inj_recent = inj_injury[(inj_injury["year"] >= 2009) & (inj_injury["year"] <= 2022)].copy()
stats_recent = stats[(stats["season_year"] >= 2009) & (stats["season_year"] <= 2022)].copy()

inj_recent["season_year"] = inj_recent["year"]

merged_pos = pd.merge(
    inj_recent[["Date", "player_upper", "season_year", "injury_group"]],
    stats_recent[["player_upper", "season_year", "Pos"]],
    on=["player_upper", "season_year"],
    how="left"
)
merged_pos = merged_pos.dropna(subset=["Pos"]).copy()

# positions：G/F/C
def simplify_pos(pos):
    if "G" in pos:
        return "G"
    if "C" in pos:
        return "C"
    if "F" in pos:
        return "F"
    return pos

merged_pos["Pos_simple"] = merged_pos["Pos"].apply(simplify_pos)
# standardize by position
pos_injury_crosstab = pd.crosstab(merged_pos["Pos_simple"], merged_pos["injury_group"])
pos_injury_prop = pos_injury_crosstab.div(pos_injury_crosstab.sum(axis=1), axis=0)
# draw bar plot by group, convert table into long format
plot_df = pos_injury_prop.reset_index().melt(id_vars="Pos_simple", var_name="InjuryType", value_name="Prop")

plt.figure(figsize=(12,6))
sns.barplot(
    data=plot_df,
    x="InjuryType",
    y="Prop",
    hue="Pos_simple"
)
plt.title("Proportion of Injury Types by Position (2009–2022)")
plt.xlabel("Injury Type")
plt.ylabel("Proportion within each Position")
plt.xticks(rotation=45)
plt.legend(title="Position")
plt.tight_layout()
plt.show()



# In[17]:


# =========================================
# A. Team-level injury counts (2000–2022)
# =========================================

#  injury table: inj_injury（with year, injury_group, Team）
inj_team = inj_injury.copy()
inj_team = inj_team[inj_team["year"].between(2015, 2022)]

# 1) every team-season injury number
team_season_inj = (
    inj_team
    .groupby(["Team", "year"])
    .size()
    .reset_index(name="inj_count")
)

print("eg. team-season injuries：")
print(team_season_inj.head())

#2) every team 2000–2022 average injury numbers per season
team_avg = (
    team_season_inj
    .groupby("Team")["inj_count"]
    .mean()
    .reset_index()
    .sort_values("inj_count", ascending=False)
)

print("average injury numbers per season ( top 20 )：")
print(team_avg.head(20))

# 3) top10
top10 = team_avg.head(10)

plt.figure(figsize=(10, 5))
sns.barplot(data=top10, x="Team", y="inj_count")
plt.title("Top 10 Teams by Average Injury Count per Season (2000–2022)")
plt.xlabel("Team")
plt.ylabel("Average injuries per season")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 4) Blazers's trend
blazers = team_season_inj[team_season_inj["Team"] == "Blazers"].sort_values("year")

plt.figure(figsize=(8, 4))
sns.lineplot(data=blazers, x="year", y="inj_count", marker="o")
plt.title("Blazers Injury Counts per Season (2000–2022)")
plt.xlabel("Season year")
plt.ylabel("Injury count")
plt.tight_layout()
plt.show()




# In[14]:


# ======================================================
# Q3: 5 games before injury vs earlier games
# ======================================================

inj_pre = inj_injury[(inj_injury["year"] >= 2010) & (inj_injury["year"] <= 2022)].copy()

box["GAME_DATE"] = pd.to_datetime(box["GAME_DATE"], errors="coerce")
box = box.dropna(subset=["GAME_DATE"])
box_recent = box[(box["season_year"] >= 2010) & (box["season_year"] <= 2022)].copy()

records = []
N = 500  # 
inj_pre_sample = inj_pre.head(N)

for idx, row in inj_pre_sample.iterrows():
    player = row["player_upper"]
    year = row["year"]
    inj_date = row["Date"]
    
    games_all = box_recent[
        (box_recent["player_upper"] == player) &
        (box_recent["season_year"] == year)
    ].sort_values("GAME_DATE")
    if games_all.empty:
        continue
    
    games_before = games_all[games_all["GAME_DATE"] < inj_date]
    if len(games_before) < 10:
        # if less than 10 games, there's no earlier and final 5 games
        continue
    
    pre5 = games_before.tail(5)
    early = games_before.iloc[:-5]  # earlier games
    
    pre5_means = pre5[["MIN", "FG_PCT", "PTS"]].mean()
    early_means = early[["MIN", "FG_PCT", "PTS"]].mean()
    
    records.append({
        "player_upper": player,
        "year": year,
        "inj_date": inj_date,
        "pre5_MIN": pre5_means["MIN"],
        "pre5_FG_PCT": pre5_means["FG_PCT"],
        "pre5_PTS": pre5_means["PTS"],
        "early_MIN": early_means["MIN"],
        "early_FG_PCT": early_means["FG_PCT"],
        "early_PTS": early_means["PTS"],
    })

pre_df = pd.DataFrame(records)
print("Number of injuries with enough games:", len(pre_df))


# delta = final 5 games - earlier games
pre_df["delta_MIN"] = pre_df["pre5_MIN"] - pre_df["early_MIN"]
pre_df["delta_FG_PCT"] = pre_df["pre5_FG_PCT"] - pre_df["early_FG_PCT"]
pre_df["delta_PTS"] = pre_df["pre5_PTS"] - pre_df["early_PTS"]

print("average change：")
print(pre_df[["delta_MIN", "delta_FG_PCT", "delta_PTS"]].mean())

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(pre_df["delta_MIN"], kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Change in Minutes (Last 5 vs Earlier)")
plt.xlabel("delta MIN")

plt.subplot(1, 3, 2)
sns.histplot(pre_df["delta_FG_PCT"], kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Change in FG% (Last 5 vs Earlier)")
plt.xlabel("delta FG_PCT")

plt.subplot(1, 3, 3)
sns.histplot(pre_df["delta_PTS"], kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Change in Points (Last 5 vs Earlier)")
plt.xlabel("delta PTS")

plt.tight_layout()
plt.show()


# In[15]:


# ============================================
# NBA Injury – Season-level Logistic Regression
# y = at least one injury record in this season
# X = Age, MP, G, GS, PTS, FGA, TRB, STL, BLK, PF, TOV
# 2002–2022 season
# ============================================


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve



# ---------- 3. only take 2002–2022 data----------
inj_mod = inj[inj["year"].between(2002, 2022)].copy()
stats_mod = stats[stats["season_year"].between(2002, 2022)].copy()

# ---------- 4. injury label y (season-level)----------

# for every (player, year), is there injury record
inj_season = (
    inj_mod.groupby(["player_upper", "year"])
    .size()
    .reset_index(name="inj_count")
    .rename(columns={"year": "season_year"})
)
inj_season["y_injured"] = 1  # with record-> 1

# merge injury info to season stats
merged = stats_mod.merge(
    inj_season[["player_upper", "season_year", "y_injured"]],
    on=["player_upper", "season_year"],
    how="left"
)
merged["y_injured"] = merged["y_injured"].fillna(0)  #no records-> 0

print("sample size total rows：", len(merged))
print("injuries vs non-injuries：")
print(merged["y_injured"].value_counts())

# ---------- 5. choose characteristics X ----------

# most relevant to injuries
Xvars = ["Age", "MP", "G", "GS", "PTS", "FGA", "TRB", "STL", "BLK", "PF", "TOV"]

# 
Xvars_present = [c for c in Xvars if c in merged.columns]
print("characteristics available：", Xvars_present)

# delete rows with na
model_df = merged.dropna(subset=Xvars_present + ["y_injured"])

X = model_df[Xvars_present].values
y = model_df["y_injured"].values

print("sample size for model：", len(model_df))

# ---------- 6. training/testing ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------- 7. standardize/regression ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_s, y_train)

y_pred = logreg.predict(X_test_s)
y_proba = logreg.predict_proba(X_test_s)[:, 1]

# ---------- 8.  summary ----------
print("=== Classification report ===")
print(classification_report(y_test, y_pred, digits=3))

auc = roc_auc_score(y_test, y_proba)
print("AUC:", auc)

# ---------- 9. barplot ----------
coef_df = pd.DataFrame({
    "feature": Xvars_present,
    "coef": logreg.coef_[0]
}).sort_values("coef", ascending=True)

plt.figure(figsize=(8, 6))
sns.barplot(data=coef_df, x="coef", y="feature")
plt.title("Logistic Regression Coefficients (Season-level Injury Risk)")
plt.axvline(0, color="black", linewidth=1)
plt.tight_layout()
plt.show()

# ---------- 10. ROC curve ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Season Injury (2002–2022)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[ ]:




