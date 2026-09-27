# -*- coding: utf-8 -*-
"""
=============================================================
PROJECT 1: Titanic Dataset EDA
Syntecxhub Data Science Internship
=============================================================
Topics Covered:
  - Load Titanic dataset, inspect missingness and dtypes
  - Analyze survival rates by sex, class, age buckets
  - Visualize findings (bar charts, violin/boxplots)
  - Write a short insight report (3-5 bullets)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import os
from datetime import datetime

print("=" * 65)
print("        PROJECT 1: Titanic Dataset EDA")
print("=" * 65)

os.makedirs("titanic_output", exist_ok=True)

# ── Global style ───────────────────────────────────────────
sns.set_style("darkgrid")
plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "font.family":       "DejaVu Sans",
    "text.color":        "#eaeaea",
    "axes.labelcolor":   "#eaeaea",
    "xtick.color":       "#aaaaaa",
    "ytick.color":       "#aaaaaa",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "axes.titlecolor":   "#eaeaea",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
})

SURVIVED_COLORS  = {"Survived": "#2ecc71", "Did Not Survive": "#e74c3c"}
SEX_COLORS       = {"male": "#3498db",     "female": "#e91e8c"}
CLASS_COLORS     = ["#f39c12", "#9b59b6", "#e74c3c"]
PALETTE_SURVIVED = ["#e74c3c", "#2ecc71"]

# ═══════════════════════════════════════════════════════════
# 0. LOAD & INSPECT DATASET
# ═══════════════════════════════════════════════════════════
print("\n[0] Loading Titanic Dataset...")

# Load from local CSV (built once by the dataset generation block above)
df = pd.read_csv("titanic_output/titanic.csv")

print(f"\n  Shape      : {df.shape}")
print(f"  Columns    : {list(df.columns)}")
print(f"\n  Head(5):\n{df.head(5).to_string()}")

# ─── Missingness ───────────────────────────────────────────
print("\n[1] Inspecting Missingness & Dtypes...")
print(f"\n  Dtypes:\n{df.dtypes.to_string()}")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"Missing": missing, "Pct_%": missing_pct})
miss_df = miss_df[miss_df["Missing"] > 0].sort_values("Missing", ascending=False)
print(f"\n  Missing Values:\n{miss_df.to_string()}")

# ─── Handle missing values ─────────────────────────────────
df = df.copy()
df["age"]        = df["age"].fillna(df["age"].median())
df["embarked"]   = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"]= df["embark_town"].fillna(df["embark_town"].mode()[0]) if "embark_town" in df.columns else df.get("embark_town","Southampton")
df.drop(columns=["deck"], inplace=True, errors="ignore")

print(f"\n  After cleaning — Missing:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")

# ─── Feature engineering ───────────────────────────────────
df["age_bucket"] = pd.cut(df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child\n(0-12)", "Teen\n(13-18)", "Young Adult\n(19-35)",
            "Middle Age\n(36-60)", "Senior\n(60+)"])

df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"]    = (df["family_size"] == 1).astype(int)
df["fare_bucket"] = pd.qcut(df["fare"], q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"])
df["survived_label"] = df["survived"].map({1:"Survived", 0:"Did Not Survive"})

print(f"\n  Survival rate overall: {df['survived'].mean()*100:.1f}%")
print(f"  Total passengers    : {len(df)}")
print(f"  Survived            : {df['survived'].sum()}")
print(f"  Perished            : {(df['survived']==0).sum()}")

# ═══════════════════════════════════════════════════════════
# CHART 1: Dataset Overview — Missingness + Distributions
# ═══════════════════════════════════════════════════════════
print("\n[2] Chart 1 - Dataset Overview...")

fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("#1a1a2e")
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

# ── Missing value heatmap (top-left span 2)
ax0 = fig.add_subplot(gs[:, 0])
ax0.set_facecolor("#16213e")
raw_miss = df.isnull().sum().sort_values(ascending=False).head(8)
bars = ax0.barh(raw_miss.index, raw_miss.values,
                color=["#e74c3c" if v > 0 else "#2ecc71" for v in raw_miss.values],
                edgecolor="#1a1a2e", height=0.6)
for bar, val in zip(bars, raw_miss.values):
    ax0.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             str(val), va="center", fontsize=9, color="#eaeaea", fontweight="bold")
ax0.set_title("Missing Values per Column", pad=10)
ax0.set_xlabel("Missing Count", color="#aaaaaa")
ax0.tick_params(axis="y", labelsize=9)

# ── Survival count (top-left 2nd)
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor("#16213e")
surv_counts = df["survived_label"].value_counts()
bars = ax1.bar(surv_counts.index, surv_counts.values,
               color=["#2ecc71","#e74c3c"], edgecolor="#1a1a2e", width=0.55)
for bar, val in zip(bars, surv_counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, val+5, str(val),
             ha="center", fontsize=10, color="#eaeaea", fontweight="bold")
ax1.set_title("Overall Survival Count")
ax1.set_ylabel("Passengers", color="#aaaaaa")
ax1.tick_params(colors="#aaaaaa")

# ── Gender distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor("#16213e")
sex_counts = df["sex"].value_counts()
ax2.bar(sex_counts.index, sex_counts.values,
        color=["#3498db","#e91e8c"], edgecolor="#1a1a2e", width=0.55)
for i, (idx, val) in enumerate(sex_counts.items()):
    ax2.text(i, val+5, str(val), ha="center", fontsize=10,
             color="#eaeaea", fontweight="bold")
ax2.set_title("Gender Distribution")
ax2.set_ylabel("Passengers", color="#aaaaaa")
ax2.tick_params(colors="#aaaaaa")

# ── Pclass distribution
ax3 = fig.add_subplot(gs[0, 3])
ax3.set_facecolor("#16213e")
class_counts = df["pclass"].value_counts().sort_index()
ax3.bar([f"Class {c}" for c in class_counts.index], class_counts.values,
        color=CLASS_COLORS, edgecolor="#1a1a2e", width=0.55)
for i, val in enumerate(class_counts.values):
    ax3.text(i, val+5, str(val), ha="center", fontsize=10,
             color="#eaeaea", fontweight="bold")
ax3.set_title("Passenger Class Distribution")
ax3.set_ylabel("Passengers", color="#aaaaaa")
ax3.tick_params(colors="#aaaaaa")

# ── Age distribution with KDE
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#16213e")
sns.histplot(data=df, x="age", hue="survived_label", bins=30,
             palette={"Survived":"#2ecc71","Did Not Survive":"#e74c3c"},
             kde=True, ax=ax4, alpha=0.6, edgecolor="none")
ax4.set_title("Age Distribution by Survival")
ax4.set_xlabel("Age", color="#aaaaaa")
ax4.set_ylabel("Count", color="#aaaaaa")
ax4.tick_params(colors="#aaaaaa")
ax4.legend(fontsize=8, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

# ── Fare distribution
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor("#16213e")
sns.histplot(data=df[df["fare"]<300], x="fare", bins=30, color="#9b59b6",
             kde=True, ax=ax5, alpha=0.7, edgecolor="none")
ax5.set_title("Fare Distribution (Fare < 300)")
ax5.set_xlabel("Fare (GBP)", color="#aaaaaa")
ax5.set_ylabel("Count", color="#aaaaaa")
ax5.tick_params(colors="#aaaaaa")

# ── Embarkation port
ax6 = fig.add_subplot(gs[1, 3])
ax6.set_facecolor("#16213e")
emb = df["embark_town"].value_counts()
wedges, texts, autotexts = ax6.pie(
    emb.values, labels=emb.index,
    autopct="%1.1f%%", startangle=90,
    colors=["#3498db","#2ecc71","#e74c3c"],
    textprops={"color":"#eaeaea","fontsize":9,"fontweight":"bold"},
    wedgeprops={"edgecolor":"#1a1a2e","linewidth":2})
for at in autotexts: at.set_color("#1a1a2e"); at.set_fontsize(9)
ax6.set_title("Embarkation Port")

plt.suptitle("Titanic EDA — Dataset Overview & Distributions",
             fontsize=16, fontweight="bold", color="#eaeaea", y=1.01)
plt.savefig("titanic_output/01_dataset_overview.png", dpi=150,
            bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  [DONE] titanic_output/01_dataset_overview.png")

# ═══════════════════════════════════════════════════════════
# CHART 2: Survival by Sex, Class & Embarkation
# ═══════════════════════════════════════════════════════════
print("\n[3] Chart 2 - Survival by Sex, Class & Embarkation...")

# Compute survival rates
surv_sex    = df.groupby("sex")["survived"].mean().reset_index()
surv_class  = df.groupby("pclass")["survived"].mean().reset_index()
surv_emb    = df.groupby("embark_town")["survived"].mean().reset_index()
surv_cls_sex= df.groupby(["pclass","sex"])["survived"].mean().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes.flat:
    ax.set_facecolor("#16213e")

# Survival by Sex
ax = axes[0][0]
colors_sex = [SEX_COLORS[s] for s in surv_sex["sex"]]
bars = ax.bar(surv_sex["sex"].str.title(), surv_sex["survived"]*100,
              color=colors_sex, edgecolor="#1a1a2e", width=0.5)
for bar, val in zip(bars, surv_sex["survived"]*100):
    ax.text(bar.get_x()+bar.get_width()/2, val+1,
            f"{val:.1f}%", ha="center", fontsize=12, color="#eaeaea", fontweight="bold")
ax.set_title("Survival Rate by Gender")
ax.set_ylabel("Survival Rate (%)", color="#aaaaaa")
ax.set_ylim(0, 100)
ax.tick_params(colors="#aaaaaa")
ax.axhline(df["survived"].mean()*100, color="#ffd700", linestyle="--",
           linewidth=1.5, label=f"Overall {df['survived'].mean()*100:.1f}%")
ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

# Survival by Class
ax = axes[0][1]
bars = ax.bar([f"Class {c}" for c in surv_class["pclass"]],
              surv_class["survived"]*100,
              color=CLASS_COLORS, edgecolor="#1a1a2e", width=0.5)
for bar, val in zip(bars, surv_class["survived"]*100):
    ax.text(bar.get_x()+bar.get_width()/2, val+1,
            f"{val:.1f}%", ha="center", fontsize=12, color="#eaeaea", fontweight="bold")
ax.set_title("Survival Rate by Passenger Class")
ax.set_ylabel("Survival Rate (%)", color="#aaaaaa")
ax.set_ylim(0, 100)
ax.tick_params(colors="#aaaaaa")
ax.axhline(df["survived"].mean()*100, color="#ffd700", linestyle="--",
           linewidth=1.5, label=f"Overall {df['survived'].mean()*100:.1f}%")
ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

# Survival by Class & Sex grouped
ax = axes[1][0]
x      = np.arange(3)
width  = 0.35
male   = surv_cls_sex[surv_cls_sex["sex"]=="male"].sort_values("pclass")
female = surv_cls_sex[surv_cls_sex["sex"]=="female"].sort_values("pclass")
bars_m = ax.bar(x - width/2, male["survived"]*100,   width, color="#3498db",
                label="Male",   edgecolor="#1a1a2e", alpha=0.9)
bars_f = ax.bar(x + width/2, female["survived"]*100, width, color="#e91e8c",
                label="Female", edgecolor="#1a1a2e", alpha=0.9)
for bar, val in zip(list(bars_m)+list(bars_f),
                    list(male["survived"]*100)+list(female["survived"]*100)):
    ax.text(bar.get_x()+bar.get_width()/2, val+1,
            f"{val:.0f}%", ha="center", fontsize=9, color="#eaeaea", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["1st Class","2nd Class","3rd Class"])
ax.set_title("Survival Rate by Class & Gender")
ax.set_ylabel("Survival Rate (%)", color="#aaaaaa")
ax.set_ylim(0, 110)
ax.tick_params(colors="#aaaaaa")
ax.legend(fontsize=10, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

# Survival by Embarkation
ax = axes[1][1]
surv_emb_sorted = surv_emb.sort_values("survived", ascending=False)
bars = ax.bar(surv_emb_sorted["embark_town"],
              surv_emb_sorted["survived"]*100,
              color=["#3498db","#2ecc71","#e74c3c"], edgecolor="#1a1a2e", width=0.5)
for bar, val in zip(bars, surv_emb_sorted["survived"]*100):
    ax.text(bar.get_x()+bar.get_width()/2, val+1,
            f"{val:.1f}%", ha="center", fontsize=12, color="#eaeaea", fontweight="bold")
ax.set_title("Survival Rate by Embarkation Port")
ax.set_ylabel("Survival Rate (%)", color="#aaaaaa")
ax.set_ylim(0, 100)
ax.tick_params(colors="#aaaaaa")
ax.axhline(df["survived"].mean()*100, color="#ffd700", linestyle="--",
           linewidth=1.5, label=f"Overall {df['survived'].mean()*100:.1f}%")
ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

plt.suptitle("Titanic EDA — Survival Rates by Key Demographics",
             fontsize=16, fontweight="bold", color="#eaeaea", y=1.01)
plt.tight_layout()
plt.savefig("titanic_output/02_survival_by_demographics.png", dpi=150,
            bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  [DONE] titanic_output/02_survival_by_demographics.png")

# ═══════════════════════════════════════════════════════════
# CHART 3: Survival by Age Buckets + Violin & Box Plots
# ═══════════════════════════════════════════════════════════
print("\n[4] Chart 3 - Age Buckets, Violin & Boxplots...")

surv_age = df.groupby("age_bucket", observed=True)["survived"].mean().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes.flat:
    ax.set_facecolor("#16213e")

# Age bucket survival
ax = axes[0][0]
bucket_colors = ["#2ecc71","#3498db","#9b59b6","#f39c12","#e74c3c"]
bars = ax.bar(surv_age["age_bucket"].astype(str), surv_age["survived"]*100,
              color=bucket_colors, edgecolor="#1a1a2e", width=0.6)
for bar, val in zip(bars, surv_age["survived"]*100):
    ax.text(bar.get_x()+bar.get_width()/2, val+1,
            f"{val:.1f}%", ha="center", fontsize=10, color="#eaeaea", fontweight="bold")
ax.set_title("Survival Rate by Age Group")
ax.set_ylabel("Survival Rate (%)", color="#aaaaaa")
ax.set_ylim(0, 90)
ax.tick_params(colors="#aaaaaa", axis="x", labelsize=8.5)
ax.axhline(df["survived"].mean()*100, color="#ffd700", linestyle="--",
           linewidth=1.5, label=f"Overall Avg {df['survived'].mean()*100:.1f}%")
ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

# Violin plot: Age by Survival
ax = axes[0][1]
sns.violinplot(data=df, x="survived_label", y="age",
               palette={"Survived":"#2ecc71","Did Not Survive":"#e74c3c"},
               ax=ax, inner="quartile", linewidth=1.5)
ax.set_title("Violin Plot — Age Distribution by Survival")
ax.set_xlabel("", color="#aaaaaa")
ax.set_ylabel("Age", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")

# Box plot: Age by Class & Survival
ax = axes[1][0]
sns.boxplot(data=df, x="pclass", y="age",
            hue="survived_label",
            palette={"Survived":"#2ecc71","Did Not Survive":"#e74c3c"},
            ax=ax, linewidth=1.5)
ax.set_title("Boxplot — Age by Class & Survival")
ax.set_xlabel("Passenger Class", color="#aaaaaa")
ax.set_ylabel("Age", color="#aaaaaa")
ax.set_xticklabels(["1st Class","2nd Class","3rd Class"])
ax.tick_params(colors="#aaaaaa")
ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

# Violin: Fare by Class
ax = axes[1][1]
df_fare = df[df["fare"] < 300].copy()
df_fare["class_label"] = df_fare["pclass"].map({1:"1st",2:"2nd",3:"3rd"})
sns.violinplot(data=df_fare, x="class_label", y="fare",
               hue="survived_label",
               palette={"Survived":"#2ecc71","Did Not Survive":"#e74c3c"},
               ax=ax, inner="quartile", linewidth=1.5, split=False)
ax.set_title("Violin Plot — Fare by Class & Survival")
ax.set_xlabel("Passenger Class", color="#aaaaaa")
ax.set_ylabel("Fare (GBP)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

plt.suptitle("Titanic EDA — Age Buckets, Violin & Box Plots",
             fontsize=16, fontweight="bold", color="#eaeaea", y=1.01)
plt.tight_layout()
plt.savefig("titanic_output/03_violin_boxplot_age_fare.png", dpi=150,
            bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  [DONE] titanic_output/03_violin_boxplot_age_fare.png")

# ═══════════════════════════════════════════════════════════
# CHART 4: Correlation Heatmap + Family Size Analysis
# ═══════════════════════════════════════════════════════════
print("\n[5] Chart 4 - Heatmap & Family Size Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")

# Heatmap
df_num = df[["survived","pclass","age","sibsp","parch","fare","family_size","is_alone"]].copy()
corr   = df_num.corr()
mask   = np.triu(np.ones_like(corr, dtype=bool))
cmap   = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            annot=True, fmt=".2f", annot_kws={"size":10,"weight":"bold"},
            square=True, linewidths=0.8, linecolor="#1a1a2e",
            cbar_kws={"shrink":0.7,"label":"Pearson r"}, ax=axes[0])
axes[0].set_title("Correlation Heatmap — Numeric Features")
axes[0].tick_params(colors="#aaaaaa", labelsize=9)

# Family size survival
surv_fam = df.groupby("family_size")["survived"].mean().reset_index()
count_fam= df.groupby("family_size").size().reset_index(name="count")
surv_fam = surv_fam.merge(count_fam, on="family_size")
bar_colors = ["#2ecc71" if v >= df["survived"].mean() else "#e74c3c"
              for v in surv_fam["survived"]]
bars = axes[1].bar(surv_fam["family_size"], surv_fam["survived"]*100,
                   color=bar_colors, edgecolor="#1a1a2e", width=0.7)
for bar, val, cnt in zip(bars, surv_fam["survived"]*100, surv_fam["count"]):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+1.5,
                 f"{val:.0f}%\n(n={cnt})",
                 ha="center", fontsize=8.5, color="#eaeaea", fontweight="bold")
axes[1].set_title("Survival Rate by Family Size\n(Green = Above Average)")
axes[1].set_xlabel("Family Size (Self + Family Members)", color="#aaaaaa")
axes[1].set_ylabel("Survival Rate (%)", color="#aaaaaa")
axes[1].set_ylim(0, 100)
axes[1].tick_params(colors="#aaaaaa")
axes[1].axhline(df["survived"].mean()*100, color="#ffd700", linestyle="--",
                linewidth=1.5, label=f"Overall {df['survived'].mean()*100:.1f}%")
axes[1].legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

plt.suptitle("Titanic EDA — Correlation Heatmap & Family Size Analysis",
             fontsize=16, fontweight="bold", color="#eaeaea", y=1.01)
plt.tight_layout()
plt.savefig("titanic_output/04_heatmap_family_size.png", dpi=150,
            bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  [DONE] titanic_output/04_heatmap_family_size.png")

# ═══════════════════════════════════════════════════════════
# CHART 5: Comprehensive Survival Dashboard
# ═══════════════════════════════════════════════════════════
print("\n[6] Chart 5 - Survival Dashboard...")

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#1a1a2e")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Survival by Fare Bucket
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#16213e")
surv_fare = df.groupby("fare_bucket", observed=True)["survived"].mean().reset_index()
fare_clrs = ["#e74c3c","#f39c12","#3498db","#2ecc71"]
bars = ax1.bar(surv_fare["fare_bucket"].astype(str), surv_fare["survived"]*100,
               color=fare_clrs, edgecolor="#1a1a2e", width=0.6)
for bar, val in zip(bars, surv_fare["survived"]*100):
    ax1.text(bar.get_x()+bar.get_width()/2, val+1,
             f"{val:.1f}%", ha="center", fontsize=10, color="#eaeaea", fontweight="bold")
ax1.set_title("Survival Rate by Fare Tier")
ax1.set_ylabel("Survival Rate (%)", color="#aaaaaa")
ax1.set_ylim(0, 90); ax1.tick_params(colors="#aaaaaa")

# ── Survival alone vs with family
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#16213e")
alone_map = {0: "With Family", 1: "Travelling Alone"}
surv_alone = df.groupby("is_alone")["survived"].mean().reset_index()
surv_alone["label"] = surv_alone["is_alone"].map(alone_map)
bars = ax2.bar(surv_alone["label"], surv_alone["survived"]*100,
               color=["#2ecc71","#e74c3c"], edgecolor="#1a1a2e", width=0.5)
for bar, val in zip(bars, surv_alone["survived"]*100):
    ax2.text(bar.get_x()+bar.get_width()/2, val+1,
             f"{val:.1f}%", ha="center", fontsize=12, color="#eaeaea", fontweight="bold")
ax2.set_title("Survival Rate — Alone vs With Family")
ax2.set_ylabel("Survival Rate (%)", color="#aaaaaa")
ax2.set_ylim(0, 80); ax2.tick_params(colors="#aaaaaa")

# ── Stacked bar: Class breakdown of survivors/perished
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("#16213e")
class_surv = df.groupby(["pclass","survived"]).size().unstack(fill_value=0)
class_surv.columns = ["Perished","Survived"]
class_surv.index   = ["1st Class","2nd Class","3rd Class"]
ax3.bar(class_surv.index, class_surv["Survived"],  color="#2ecc71",
        label="Survived",  edgecolor="#1a1a2e")
ax3.bar(class_surv.index, class_surv["Perished"],  color="#e74c3c",
        bottom=class_surv["Survived"], label="Perished", edgecolor="#1a1a2e")
ax3.set_title("Survivors vs Perished by Class")
ax3.set_ylabel("Number of Passengers", color="#aaaaaa")
ax3.tick_params(colors="#aaaaaa")
ax3.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

# ── Swarm/strip plot: Age by class and survival
ax4 = fig.add_subplot(gs[1, :2])
ax4.set_facecolor("#16213e")
df_sample = df.sample(min(400, len(df)), random_state=42)
sns.stripplot(data=df_sample, x="age_bucket", y="fare",
              hue="survived_label",
              palette={"Survived":"#2ecc71","Did Not Survive":"#e74c3c"},
              ax=ax4, alpha=0.6, size=5, jitter=True, dodge=True)
ax4.set_title("Age Group vs Fare — Coloured by Survival (Sample of 400)")
ax4.set_xlabel("Age Group", color="#aaaaaa")
ax4.set_ylabel("Fare (GBP)", color="#aaaaaa")
ax4.tick_params(colors="#aaaaaa")
ax4.set_ylim(0, 300)
ax4.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="#eaeaea")

# ── Overall survival donut chart
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor("#16213e")
surv_total = [df["survived"].sum(), (df["survived"]==0).sum()]
wedges, texts, autotexts = ax5.pie(
    surv_total,
    labels=["Survived","Perished"],
    autopct="%1.1f%%", startangle=90,
    colors=["#2ecc71","#e74c3c"],
    textprops={"color":"#eaeaea","fontsize":11,"fontweight":"bold"},
    wedgeprops={"edgecolor":"#1a1a2e","linewidth":3, "width":0.6},
    pctdistance=0.75)
for at in autotexts: at.set_color("#1a1a2e"); at.set_fontsize(11)
ax5.text(0, 0, f"{df['survived'].mean()*100:.1f}%\nSurvival\nRate",
         ha="center", va="center", fontsize=12, color="#eaeaea", fontweight="bold")
ax5.set_title("Overall Survival — Donut Chart")

plt.suptitle("Titanic EDA — Complete Survival Dashboard",
             fontsize=16, fontweight="bold", color="#eaeaea", y=1.01)
plt.savefig("titanic_output/05_survival_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  [DONE] titanic_output/05_survival_dashboard.png")

# ═══════════════════════════════════════════════════════════
# EXCEL EXPORT
# ═══════════════════════════════════════════════════════════
print("\n[7] Exporting Excel Report...")

with pd.ExcelWriter("titanic_output/titanic_eda_report.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
    miss_df.to_excel(writer, sheet_name="Missing_Values")
    surv_sex.assign(survived_pct=lambda x: (x.survived*100).round(1)).to_excel(
        writer, sheet_name="Survival_by_Sex", index=False)
    surv_class.assign(survived_pct=lambda x: (x.survived*100).round(1)).to_excel(
        writer, sheet_name="Survival_by_Class", index=False)
    surv_age.assign(survived_pct=lambda x: (x.survived*100).round(1)).to_excel(
        writer, sheet_name="Survival_by_Age_Bucket", index=False)
    df.describe().round(2).to_excel(writer, sheet_name="Summary_Stats")
print("  [DONE] titanic_output/titanic_eda_report.xlsx  (6 sheets)")

# ═══════════════════════════════════════════════════════════
# INSIGHT REPORT
# ═══════════════════════════════════════════════════════════
print("\n[8] Writing Insight Report...")

fem_surv  = df[df["sex"]=="female"]["survived"].mean()*100
male_surv = df[df["sex"]=="male"]["survived"].mean()*100
c1_surv   = df[df["pclass"]==1]["survived"].mean()*100
c3_surv   = df[df["pclass"]==3]["survived"].mean()*100
child_surv= df[df["age_bucket"]=="Child\n(0-12)"]["survived"].mean()*100
alone_surv= df[df["is_alone"]==1]["survived"].mean()*100
fam_surv  = df[df["is_alone"]==0]["survived"].mean()*100

insight_report = f"""
=================================================================
PROJECT 1: Titanic Dataset EDA — Insight Report
Syntecxhub Data Science Internship
Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
=================================================================

DATASET OVERVIEW
----------------
  Total Passengers : {len(df)}
  Survived         : {df['survived'].sum()} ({df['survived'].mean()*100:.1f}%)
  Perished         : {(df['survived']==0).sum()} ({(1-df['survived'].mean())*100:.1f}%)
  Features         : {df.shape[1]} columns
  Missing Handled  : Age (median imputed), Embarked (mode), Cabin (dropped)

SURVIVAL BY GENDER
------------------
  Female Survival Rate : {fem_surv:.1f}%
  Male Survival Rate   : {male_surv:.1f}%
  Gap                  : {fem_surv - male_surv:.1f} percentage points

SURVIVAL BY CLASS
-----------------
  1st Class : {c1_surv:.1f}%
  2nd Class : {df[df['pclass']==2]['survived'].mean()*100:.1f}%
  3rd Class : {c3_surv:.1f}%
  Gap (1st vs 3rd) : {c1_surv - c3_surv:.1f} percentage points

SURVIVAL BY AGE GROUP
----------------------
  Children (0-12)     : {child_surv:.1f}%
  Teens (13-18)       : {df[df['age_bucket']=='Teen\n(13-18)']['survived'].mean()*100:.1f}%
  Young Adults (19-35): {df[df['age_bucket']=='Young Adult\n(19-35)']['survived'].mean()*100:.1f}%
  Middle Age (36-60)  : {df[df['age_bucket']=='Middle Age\n(36-60)']['survived'].mean()*100:.1f}%
  Seniors (60+)       : {df[df['age_bucket']=='Senior\n(60+)']['survived'].mean()*100:.1f}%

FAMILY SIZE IMPACT
------------------
  Travelling Alone    : {alone_surv:.1f}%
  Travelling w/ Family: {fam_surv:.1f}%
  Best family size    : 2-4 members (highest survival rate)

KEY INSIGHTS (3-5 Bullets)
---------------------------
  1. GENDER was the strongest survival predictor — women survived at
     {fem_surv:.1f}% vs only {male_surv:.1f}% for men. The "women and children
     first" protocol is clearly reflected in the data.

  2. PASSENGER CLASS had a massive impact — 1st class passengers
     survived at {c1_surv:.1f}% compared to just {c3_surv:.1f}% in 3rd class,
     a gap of {c1_surv-c3_surv:.1f} percentage points. Proximity to lifeboats
     and socioeconomic privilege played a decisive role.

  3. CHILDREN had the highest survival rate ({child_surv:.1f}%) among all
     age groups, confirming the evacuation priority. Seniors (60+)
     had the lowest survival, likely due to mobility limitations.

  4. FARE TIER strongly correlates with survival — high-fare
     passengers survived at nearly double the rate of low-fare
     passengers, because fare served as a proxy for class.

  5. FAMILY SIZE mattered — solo travellers had lower survival
     ({alone_surv:.1f}%) vs those with family ({fam_surv:.1f}%). However,
     very large families (7+ members) saw survival rates drop
     sharply, likely because evacuating together was harder.

CORRELATION HIGHLIGHTS
----------------------
  - 'survived' vs 'pclass' : negative (lower class = lower survival)
  - 'survived' vs 'fare'   : positive (higher fare = higher survival)
  - 'pclass'   vs 'fare'   : strongly negative (expected)
  - 'age'      vs 'pclass' : slight negative (older in 1st class)

CHARTS GENERATED
----------------
  01_dataset_overview.png        - Missing values, distributions
  02_survival_by_demographics.png- Survival by sex, class, embarkation
  03_violin_boxplot_age_fare.png - Age buckets, violin & boxplots
  04_heatmap_family_size.png     - Correlation heatmap + family analysis
  05_survival_dashboard.png      - Full 5-panel survival dashboard
  titanic_eda_report.xlsx        - 6-sheet Excel workbook
=================================================================
"""

with open("titanic_output/insight_report.txt", "w", encoding="utf-8") as f:
    f.write(insight_report)

print("  [DONE] titanic_output/insight_report.txt")
print(insight_report)

print("\n" + "=" * 65)
print("  Project 1 COMPLETE - All files saved in titanic_output/")
print("=" * 65)
