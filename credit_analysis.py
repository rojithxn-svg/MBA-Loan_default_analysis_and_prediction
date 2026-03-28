"""
=============================================================================
  CREDIT RISK SCORING & LOAN DEFAULT PREDICTION
  Full End-to-End Analysis — MBA Final Project
  Author  : Rojith Nesar XN | Reg No: 69123200139
  Dataset : Give Me Some Credit (Kaggle)

  HOW TO RUN:
    1. Place this file and cs-training.csv in the SAME folder
    2. Open Terminal / Command Prompt in that folder
    3. Run:  python credit_risk_full_analysis.py
    4. All charts saved to:   outputs/charts/
    5. All results saved to:  outputs/results/
    6. Upload the entire outputs/ folder back to Claude

  PACKAGES NEEDED (run once):
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow scipy openpyxl
=============================================================================
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, json, time
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics         import (roc_auc_score, roc_curve, auc,
                                     classification_report, confusion_matrix,
                                     f1_score, precision_score, recall_score,
                                     average_precision_score, precision_recall_curve)
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.inspection      import permutation_importance
from sklearn.neural_network  import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("  [OK] XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  [WARN] XGBoost not installed — skipping. Run: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    print("  [OK] TensorFlow available —", tf.__version__)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("  [WARN] TensorFlow not installed — will use sklearn MLP instead. Run: pip install tensorflow")

warnings.filterwarnings('ignore')
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

# ─── OUTPUT DIRECTORIES ───────────────────────────────────────────────────────
os.makedirs('outputs/charts',  exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

# ─── CHART STYLE ─────────────────────────────────────────────────────────────
C = {
    'navy'   : '#1F3864',
    'blue'   : '#2E74B5',
    'red'    : '#C00000',
    'green'  : '#375623',
    'amber'  : '#E6A817',
    'purple' : '#7030A0',
    'teal'   : '#008080',
    'grey'   : '#555555',
    'light'  : '#EBF3FB',
    'grid'   : '#E0E0E0',
}
MODEL_COLORS = {
    'Logistic Regression' : C['blue'],
    'Random Forest'       : C['green'],
    'Gradient Boosting'   : C['amber'],
    'XGBoost'             : C['red'],
    'Neural Network'      : C['purple'],
}

plt.rcParams.update({
    'figure.dpi'        : 150,
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.color'        : C['grid'],
    'grid.linewidth'    : 0.5,
    'axes.labelsize'    : 11,
    'axes.titlesize'    : 12,
    'xtick.labelsize'   : 10,
    'ytick.labelsize'   : 10,
})

def save_fig(fig, name):
    path = f"outputs/charts/{name}.png"
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f"    [SAVED] {path}")

def section(num, title):
    print(f"\n{'='*65}")
    print(f"  SECTION {num}: {title}")
    print(f"{'='*65}")

# ─── KS STATISTIC ────────────────────────────────────────────────────────────
def ks_stat(y_true, y_prob):
    df = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p', ascending=False)
    df['cum_bad']  = df['y'].cumsum()     / max(df['y'].sum(), 1)
    df['cum_good'] = (1-df['y']).cumsum() / max((1-df['y']).sum(), 1)
    return (df['cum_bad'] - df['cum_good']).abs().max()

def gini(auc_val):
    return 2 * auc_val - 1


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
section(1, "DATA LOADING & PREPARATION")

print("\n  Loading cs-training.csv ...")
try:
    df_raw = pd.read_csv('cs-training.csv')
except FileNotFoundError:
    print("\n  ERROR: cs-training.csv not found in this folder!")
    print("  Please download from: https://www.kaggle.com/competitions/GiveMeSomeCredit/data")
    print("  and place it in the same folder as this script.\n")
    exit(1)

# Rename columns
df_raw.rename(columns={
    'SeriousDlqin2yrs'                      : 'default',
    'RevolvingUtilizationOfUnsecuredLines'   : 'credit_utilization',
    'age'                                    : 'age',
    'NumberOfTime30-59DaysPastDueNotWorse'   : 'delinquency_30_59',
    'DebtRatio'                              : 'debt_to_income',
    'MonthlyIncome'                          : 'monthly_income',
    'NumberOfOpenCreditLinesAndLoans'        : 'num_accounts',
    'NumberOfTimes90DaysLate'                : 'delinquency_90_plus',
    'NumberRealEstateLoansOrLines'           : 'real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse'   : 'delinquency_60_89',
    'NumberOfDependents'                     : 'dependents',
}, inplace=True)

# Drop ID column
df_raw = df_raw.drop(columns=['Unnamed: 0'], errors='ignore')

# Remove rows with missing target
df_raw = df_raw.dropna(subset=['default'])
df_raw['default'] = df_raw['default'].astype(int)

print(f"  Raw dataset shape   : {df_raw.shape}")
print(f"  Default rate        : {df_raw['default'].mean():.2%}")
print(f"  Non-default records : {(df_raw['default']==0).sum():,}")
print(f"  Default records     : {(df_raw['default']==1).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA INTEGRITY AUDIT
# ══════════════════════════════════════════════════════════════════════════════
section(2, "DATA INTEGRITY AUDIT")

# Clip extreme outliers before audit
df_raw['credit_utilization'] = df_raw['credit_utilization'].clip(0, 1)
df_raw['debt_to_income']     = df_raw['debt_to_income'].clip(0, 5)
df_raw['age']                = df_raw['age'].clip(18, 95)

audit = pd.DataFrame({
    'dtype'         : df_raw.dtypes,
    'non_null'      : df_raw.notnull().sum(),
    'missing'       : df_raw.isnull().sum(),
    'missing_pct'   : (df_raw.isnull().mean() * 100).round(2),
    'unique_values' : df_raw.nunique(),
    'mean'          : df_raw.mean(numeric_only=True).round(3),
    'std'           : df_raw.std(numeric_only=True).round(3),
    'min'           : df_raw.min(numeric_only=True).round(3),
    'max'           : df_raw.max(numeric_only=True).round(3),
})
print("\n  DATA AUDIT TABLE:")
print("  " + "-"*70)
print(audit.to_string())
audit.to_csv('outputs/results/01_data_audit.csv')
df_raw.describe().round(3).to_csv('outputs/results/02_descriptive_stats.csv')
print("\n  [SAVED] outputs/results/01_data_audit.csv")
print("  [SAVED] outputs/results/02_descriptive_stats.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════════════
section(3, "EXPLORATORY DATA ANALYSIS")

NUM_COLS = ['credit_utilization', 'age', 'num_accounts',
            'delinquency_30_59', 'debt_to_income', 'monthly_income',
            'delinquency_90_plus', 'real_estate_loans',
            'delinquency_60_89', 'dependents']

# ── EDA CHART 1: Target Distribution ─────────────────────────────────────────
print("\n  [EDA 1] Target variable distribution...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 1: Loan Default — Target Variable Distribution',
             fontsize=13, fontweight='bold', color=C['navy'])

vc = df_raw['default'].value_counts()
axes[0].pie([vc[0], vc[1]],
            labels=[f'Non-Default\n{vc[0]:,} ({vc[0]/len(df_raw):.1%})',
                    f'Default\n{vc[1]:,} ({vc[1]/len(df_raw):.1%})'],
            colors=[C['blue'], C['red']], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11},
            explode=(0, 0.05))
axes[0].set_title('Overall Class Distribution', fontweight='bold', color=C['navy'])

axes[1].bar(['Non-Default', 'Default'], [vc[0], vc[1]],
            color=[C['blue'], C['red']], alpha=0.85, edgecolor='white', width=0.5)
axes[1].set_title('Absolute Count Comparison', fontweight='bold', color=C['navy'])
axes[1].set_ylabel('Number of Records')
for i, v in enumerate([vc[0], vc[1]]):
    axes[1].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
save_fig(fig, '01_target_distribution')

# ── EDA CHART 2: Missing Value Analysis ──────────────────────────────────────
print("  [EDA 2] Missing value analysis...")
missing = df_raw.isnull().mean() * 100
missing = missing[missing > 0].sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle('Figure 2: Missing Value Analysis by Feature',
             fontsize=13, fontweight='bold', color=C['navy'])
bars = ax.barh(missing.index, missing.values, color=C['red'], alpha=0.75, edgecolor='white')
for bar, val in zip(bars, missing.values):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=10, color=C['navy'])
ax.set_xlabel('Missing Value Percentage (%)')
ax.set_title('Only columns with missing values shown', color=C['grey'], fontsize=10)
plt.tight_layout()
save_fig(fig, '02_missing_values')
missing.to_csv('outputs/results/03_missing_values.csv')

# ── EDA CHART 3: Distribution of All Numerical Features ──────────────────────
print("  [EDA 3] Numerical feature distributions...")
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('Figure 3: Distribution of All Numerical Features (by Default Status)',
             fontsize=13, fontweight='bold', color=C['navy'])
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    clip_val = df_raw[col].quantile(0.98)
    data_clip = df_raw[col].clip(upper=clip_val)
    axes[i].hist(data_clip[df_raw['default']==0], bins=40, alpha=0.6,
                 color=C['blue'], label='Non-Default', density=True)
    axes[i].hist(data_clip[df_raw['default']==1], bins=40, alpha=0.6,
                 color=C['red'], label='Default', density=True)
    axes[i].set_title(col.replace('_', ' ').title(), fontweight='bold',
                      color=C['navy'], fontsize=9)
    axes[i].set_xlabel('')
    if i == 0: axes[i].legend(fontsize=8)
plt.tight_layout()
save_fig(fig, '03_feature_distributions')

# ── EDA CHART 4: Correlation Heatmap ─────────────────────────────────────────
print("  [EDA 4] Correlation heatmap...")
corr_df = df_raw[NUM_COLS + ['default']].copy()
# Impute for correlation only
for col in corr_df.columns:
    corr_df[col] = corr_df[col].fillna(corr_df[col].median())
corr_matrix = corr_df.corr()
corr_matrix.to_csv('outputs/results/04_correlation_matrix.csv')

fig, ax = plt.subplots(figsize=(12, 9))
fig.suptitle('Figure 4: Pearson Correlation Heatmap — All Features vs Default',
             fontsize=13, fontweight='bold', color=C['navy'])
mask = np.zeros_like(corr_matrix, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 8}, square=True,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8})
ax.tick_params(axis='x', rotation=40, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)
plt.tight_layout()
save_fig(fig, '04_correlation_heatmap')

# ── EDA CHART 5: Default Rate by Feature Quartile ────────────────────────────
print("  [EDA 5] Default rate by feature quartile...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Figure 5: Default Rate (%) by Feature Quartile — Key Risk Drivers',
             fontsize=13, fontweight='bold', color=C['navy'])
axes = axes.flatten()
key_features = ['credit_utilization', 'debt_to_income', 'age',
                 'delinquency_30_59', 'monthly_income', 'num_accounts']

quartile_summary = {}
for i, col in enumerate(key_features):
    tmp = df_raw[[col, 'default']].dropna()
    tmp['quartile'] = pd.qcut(tmp[col], q=4, duplicates='drop')
    dr = tmp.groupby('quartile', observed=True)['default'].mean() * 100
    quartile_summary[col] = dr.to_dict()
    bar_colors = [C['red'] if v > 15 else C['blue'] for v in dr.values]
    axes[i].bar(range(len(dr)), dr.values, color=bar_colors, alpha=0.85, edgecolor='white')
    axes[i].set_xticks(range(len(dr)))
    axes[i].set_xticklabels([f'Q{j+1}' for j in range(len(dr))], fontsize=9)
    axes[i].set_title(col.replace('_',' ').title(), fontweight='bold', color=C['navy'])
    axes[i].set_ylabel('Default Rate (%)')
    for j, (bar, val) in enumerate(zip(axes[i].patches, dr.values)):
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

pd.DataFrame(quartile_summary).to_csv('outputs/results/05_default_rate_by_quartile.csv')
plt.tight_layout()
save_fig(fig, '05_default_rate_by_quartile')

# ── EDA CHART 6: Outlier Analysis (Boxplots) ─────────────────────────────────
print("  [EDA 6] Outlier boxplot analysis...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Figure 6: Outlier Analysis — Boxplots by Default Status',
             fontsize=13, fontweight='bold', color=C['navy'])
axes = axes.flatten()
outlier_cols = ['credit_utilization', 'debt_to_income', 'monthly_income',
                'age', 'num_accounts', 'real_estate_loans']
for i, col in enumerate(outlier_cols):
    clip_pct = df_raw[col].quantile(0.95)
    d0 = df_raw[df_raw['default']==0][col].dropna().clip(upper=clip_pct)
    d1 = df_raw[df_raw['default']==1][col].dropna().clip(upper=clip_pct)
    axes[i].boxplot([d0, d1],
                    labels=['Non-Default', 'Default'],
                    patch_artist=True,
                    boxprops=dict(facecolor=C['light']),
                    medianprops=dict(color=C['red'], linewidth=2),
                    whiskerprops=dict(color=C['navy']),
                    capprops=dict(color=C['navy']),
                    flierprops=dict(marker='o', markerfacecolor=C['grey'],
                                    markersize=2, alpha=0.3))
    axes[i].set_title(col.replace('_',' ').title(), fontweight='bold', color=C['navy'])
    axes[i].set_ylabel('Value (clipped at 95th pct)')
plt.tight_layout()
save_fig(fig, '06_outlier_boxplots')

# ── EDA CHART 7: Delinquency Deep-Dive ───────────────────────────────────────
print("  [EDA 7] Delinquency analysis...")
df_raw['total_delinquencies'] = (df_raw['delinquency_30_59'].fillna(0) +
                                  df_raw['delinquency_60_89'].fillna(0) +
                                  df_raw['delinquency_90_plus'].fillna(0))
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 7: Delinquency History — Relationship with Default',
             fontsize=13, fontweight='bold', color=C['navy'])

for ax, col, title in zip(axes,
    ['delinquency_30_59', 'delinquency_90_plus', 'total_delinquencies'],
    ['30–59 Days Past Due', '90+ Days Past Due', 'Total Delinquencies']):
    dr = df_raw.groupby(col.split('.')[0])['default'].mean() * 100
    dr = dr[dr.index <= 8]
    bar_c = [C['red'] if v > 20 else C['blue'] for v in dr.values]
    ax.bar(dr.index, dr.values, color=bar_c, alpha=0.85, edgecolor='white')
    ax.set_title(title, fontweight='bold', color=C['navy'])
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Default Rate (%)')
    for xi, yi in zip(dr.index, dr.values):
        ax.text(xi, yi + 0.5, f'{yi:.0f}%', ha='center', fontsize=8)

plt.tight_layout()
save_fig(fig, '07_delinquency_analysis')

# ── EDA CHART 8: Age & Income Segmentation ───────────────────────────────────
print("  [EDA 8] Age and income segmentation...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 8: Age and Income Segmentation vs Default Rate',
             fontsize=13, fontweight='bold', color=C['navy'])

df_raw['age_band'] = pd.cut(df_raw['age'],
    bins=[17, 25, 35, 45, 55, 65, 100],
    labels=['18–25', '26–35', '36–45', '46–55', '56–65', '65+'])
age_dr = df_raw.groupby('age_band', observed=True)['default'].mean() * 100
axes[0].bar(age_dr.index, age_dr.values,
            color=[C['red'] if v > 8 else C['blue'] for v in age_dr.values],
            alpha=0.85, edgecolor='white')
axes[0].set_title('Default Rate by Age Band', fontweight='bold', color=C['navy'])
axes[0].set_xlabel('Age Band')
axes[0].set_ylabel('Default Rate (%)')
for i, v in enumerate(age_dr.values):
    axes[0].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=9)

income_clip = df_raw['monthly_income'].clip(upper=df_raw['monthly_income'].quantile(0.95))
df_raw['income_band'] = pd.qcut(income_clip.fillna(income_clip.median()).rank(method='first'),
    q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
inc_dr = df_raw.groupby('income_band', observed=True)['default'].mean() * 100
axes[1].bar(inc_dr.index, inc_dr.values,
            color=[C['red'] if v > 8 else C['blue'] for v in inc_dr.values],
            alpha=0.85, edgecolor='white')
axes[1].set_title('Default Rate by Income Band', fontweight='bold', color=C['navy'])
axes[1].set_xlabel('Income Band (Quintile)')
axes[1].set_ylabel('Default Rate (%)')
for i, v in enumerate(inc_dr.values):
    axes[1].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
save_fig(fig, '08_age_income_segmentation')

# Save EDA summary
eda_summary = {
    'total_records'       : int(len(df_raw)),
    'default_count'       : int(df_raw['default'].sum()),
    'non_default_count'   : int((df_raw['default']==0).sum()),
    'default_rate'        : float(df_raw['default'].mean()),
    'imbalance_ratio'     : float((df_raw['default']==0).sum() / df_raw['default'].sum()),
    'missing_pct_income'  : float(df_raw['monthly_income'].isnull().mean()),
    'missing_pct_depend'  : float(df_raw['dependents'].isnull().mean()),
    'corr_util_default'   : float(corr_matrix.loc['credit_utilization','default']),
    'corr_dti_default'    : float(corr_matrix.loc['debt_to_income','default']),
    'corr_age_default'    : float(corr_matrix.loc['age','default']),
}
with open('outputs/results/06_eda_summary.json', 'w') as f:
    json.dump(eda_summary, f, indent=2)
print("  [SAVED] outputs/results/06_eda_summary.json")
print("\n  ✅  EDA complete — 8 charts generated")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
section(4, "FEATURE ENGINEERING")

df = df_raw.copy()

# Impute missing values
df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())
df['dependents']     = df['dependents'].fillna(df['dependents'].median())

# Engineered features
df['total_delinquencies']  = (df['delinquency_30_59'].fillna(0) +
                               df['delinquency_60_89'].fillna(0) +
                               df['delinquency_90_plus'].fillna(0))
df['high_util_flag']       = (df['credit_utilization'] > 0.75).astype(int)
df['repeat_delinquent']    = (df['total_delinquencies'] >= 3).astype(int)
df['young_borrower']       = (df['age'] < 30).astype(int)
df['senior_borrower']      = (df['age'] > 60).astype(int)
df['income_per_account']   = df['monthly_income'] / (df['num_accounts'].clip(1))
df['debt_util_interaction'] = df['debt_to_income'] * df['credit_utilization']

FEATURES = [
    'credit_utilization', 'age', 'num_accounts',
    'delinquency_30_59', 'debt_to_income', 'monthly_income',
    'delinquency_90_plus', 'real_estate_loans',
    'delinquency_60_89', 'dependents',
    'total_delinquencies', 'high_util_flag', 'repeat_delinquent',
    'young_borrower', 'senior_borrower',
    'income_per_account', 'debt_util_interaction',
]
TARGET = 'default'

print(f"  Total features after engineering: {len(FEATURES)}")
for col in FEATURES:
    miss = df[col].isnull().sum()
    if miss > 0:
        df[col] = df[col].fillna(df[col].median())

feat_df = pd.DataFrame({
    'feature'    : FEATURES,
    'default_rate_high_quartile' : [
        df[df[col] >= df[col].quantile(0.75)]['default'].mean() * 100
        for col in FEATURES
    ],
    'default_rate_low_quartile'  : [
        df[df[col] <= df[col].quantile(0.25)]['default'].mean() * 100
        for col in FEATURES
    ],
})
feat_df['spread'] = feat_df['default_rate_high_quartile'] - feat_df['default_rate_low_quartile']
feat_df = feat_df.sort_values('spread', ascending=False)
feat_df.to_csv('outputs/results/07_feature_engineering.csv', index=False)
print("  [SAVED] outputs/results/07_feature_engineering.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PREPROCESSING & SPLITS
# ══════════════════════════════════════════════════════════════════════════════
section(5, "PREPROCESSING AND DATA SPLITS")

X = df[FEATURES].values
y = df[TARGET].values

# Chronological OOT split (last 15%)
n_total    = len(X)
oot_start  = int(n_total * 0.85)
X_model, y_model = X[:oot_start], y[:oot_start]
X_oot,   y_oot   = X[oot_start:], y[oot_start:]

# Train/test split of modelling data (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y_model, test_size=0.20, stratify=y_model, random_state=42)

# Scale
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_oot   = scaler.transform(X_oot)

print(f"  Modelling set  : {len(X_model):,} records  | Default: {y_model.mean():.2%}")
print(f"  Training set   : {len(X_train):,} records  | Default: {y_train.mean():.2%}")
print(f"  Test set       : {len(X_test):,}  records  | Default: {y_test.mean():.2%}")
print(f"  OOT set        : {len(X_oot):,}   records  | Default: {y_oot.mean():.2%}")
print(f"  Imbalance ratio: {(y_train==0).sum()/(y_train==1).sum():.1f} : 1")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — SMOTE
# ══════════════════════════════════════════════════════════════════════════════
section(6, "CLASS IMBALANCE — SMOTE")

def smote(X, y, target_ratio=0.35, k=5, seed=42):
    from sklearn.neighbors import NearestNeighbors
    rng      = np.random.default_rng(seed)
    min_idx  = np.where(y==1)[0]
    X_min    = X[min_idx]
    n_min    = len(min_idx)
    n_maj    = (y==0).sum()
    n_needed = int(n_maj * target_ratio) - n_min
    if n_needed <= 0:
        return X, y
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_min)
    _, indices = nn.kneighbors(X_min)
    synthetic = []
    for _ in range(n_needed):
        i   = rng.integers(0, n_min)
        nbr = indices[i, rng.integers(1, k+1)]
        gap = rng.random()
        synthetic.append(X_min[i] + gap * (X_min[nbr] - X_min[i]))
    X_syn = np.array(synthetic)
    X_res = np.vstack([X, X_syn])
    y_res = np.concatenate([y, np.ones(len(X_syn), dtype=int)])
    perm  = rng.permutation(len(X_res))
    return X_res[perm], y_res[perm]

print("\n  Applying SMOTE to training set only...")
print(f"  Before: {len(X_train):,} records | Default: {y_train.mean():.2%}")
X_train_s, y_train_s = smote(X_train, y_train, target_ratio=0.35)
print(f"  After : {len(X_train_s):,} records | Default: {y_train_s.mean():.2%}")

smote_info = {
    'before_total'   : int(len(X_train)),
    'before_default' : int(y_train.sum()),
    'before_rate'    : float(y_train.mean()),
    'after_total'    : int(len(X_train_s)),
    'after_default'  : int(y_train_s.sum()),
    'after_rate'     : float(y_train_s.mean()),
}
with open('outputs/results/08_smote_info.json', 'w') as f:
    json.dump(smote_info, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
section(7, "MODEL TRAINING")

models_to_train = {
    'Logistic Regression': LogisticRegression(
        C=0.1, max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42),
}

if XGBOOST_AVAILABLE:
    scale_pw = float((y_train==0).sum() / (y_train==1).sum())
    models_to_train['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric='auc', random_state=42, verbosity=0,
        use_label_encoder=False)

# Neural Network (sklearn MLP as fallback)
models_to_train['Neural Network (MLP)'] = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    alpha=0.001, batch_size=256,
    max_iter=100, random_state=42,
    early_stopping=True, validation_fraction=0.1)

trained_models = {}
train_times    = {}

print()
for name, model in models_to_train.items():
    print(f"  Training: {name} ...", end='', flush=True)
    t0 = time.time()
    # Neural network and XGBoost train better on original (SMOTE optional)
    if 'Neural' in name or 'XGBoost' in name:
        model.fit(X_train_s, y_train_s)
    else:
        model.fit(X_train_s, y_train_s)
    elapsed = time.time() - t0
    train_times[name] = round(elapsed, 2)

    y_prob = model.predict_proba(X_test)[:,1]
    auc_val = roc_auc_score(y_test, y_prob)
    trained_models[name] = model
    print(f"  done ({elapsed:.1f}s) | Test AUC: {auc_val:.4f}")

# ── Deep Neural Network with TensorFlow (if available) ────────────────────────
if TENSORFLOW_AVAILABLE:
    print(f"\n  Training: Deep Neural Network (TensorFlow) ...")
    t0 = time.time()

    dnn = keras.Sequential([
        layers.Input(shape=(X_train_s.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])

    pos_weight = float((y_train_s==0).sum() / (y_train_s==1).sum())
    dnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC'])

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=5, restore_best_weights=True, mode='max')

    history = dnn.fit(
        X_train_s, y_train_s,
        validation_split=0.15,
        epochs=50, batch_size=512,
        class_weight={0: 1.0, 1: pos_weight},
        callbacks=[early_stop],
        verbose=0)

    elapsed = time.time() - t0
    train_times['Deep Neural Network'] = round(elapsed, 2)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('outputs/results/09_dnn_training_history.csv', index=False)

    # Wrapper class to match sklearn interface
    class KerasWrapper:
        def __init__(self, model): self.model = model
        def predict_proba(self, X):
            p = self.model.predict(X, verbose=0).flatten()
            return np.column_stack([1-p, p])

    trained_models['Deep Neural Network'] = KerasWrapper(dnn)
    y_prob_dnn = dnn.predict(X_test, verbose=0).flatten()
    auc_dnn = roc_auc_score(y_test, y_prob_dnn)
    print(f"  done ({elapsed:.1f}s) | Test AUC: {auc_dnn:.4f}")

    # Plot DNN training curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figure 9: Deep Neural Network — Training History',
                 fontsize=13, fontweight='bold', color=C['navy'])
    axes[0].plot(hist_df['loss'],     label='Training Loss',   color=C['red'])
    axes[0].plot(hist_df['val_loss'], label='Validation Loss', color=C['blue'])
    axes[0].set_title('Loss Curve', fontweight='bold', color=C['navy'])
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Binary Crossentropy Loss')
    axes[0].legend()
    axes[1].plot(hist_df['auc'],     label='Training AUC',   color=C['green'])
    axes[1].plot(hist_df['val_auc'], label='Validation AUC', color=C['blue'])
    axes[1].set_title('AUC Curve', fontweight='bold', color=C['navy'])
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC-ROC')
    axes[1].legend()
    plt.tight_layout()
    save_fig(fig, '09_dnn_training_history')


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
section(8, "MODEL EVALUATION AND COMPARISON")

results = {}
print()
for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob > 0.5).astype(int)

    auc_val = roc_auc_score(y_test, y_prob)
    ks_val  = ks_stat(y_test, y_prob)
    results[name] = {
        'AUC-ROC'     : round(auc_val, 4),
        'Gini'        : round(gini(auc_val), 4),
        'KS Statistic': round(ks_val, 4),
        'F1 (Default)': round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        'Precision'   : round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        'Recall'      : round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        'PR-AUC'      : round(average_precision_score(y_test, y_prob), 4),
        'Train Time(s)': train_times.get(name, 0),
    }
    print(f"  {name:<28s} AUC={auc_val:.4f}  KS={ks_val:.4f}  Gini={gini(auc_val):.4f}")

results_df = pd.DataFrame(results).T
results_df.to_csv('outputs/results/10_model_comparison_matrix.csv')
print("\n  [SAVED] outputs/results/10_model_comparison_matrix.csv")
print("\n  MODEL COMPARISON MATRIX:")
print("  " + "─"*75)
print(results_df.to_string())

# ── CHART: ROC Curves ─────────────────────────────────────────────────────────
print("\n  Generating ROC comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Figure 10: ROC Curves and Key Metrics — All Models',
             fontsize=13, fontweight='bold', color=C['navy'])

line_styles = ['-', '--', '-.', ':', '-', '--']
for i, (name, model) in enumerate(trained_models.items()):
    y_prob      = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val     = roc_auc_score(y_test, y_prob)
    color       = MODEL_COLORS.get(name.split(' ')[0] + ' ' + name.split(' ')[1]
                                   if len(name.split(' '))>1 else name,
                                   list(MODEL_COLORS.values())[i % len(MODEL_COLORS)])
    axes[0].plot(fpr, tpr, linestyle=line_styles[i % len(line_styles)],
                 color=color, linewidth=2,
                 label=f'{name} (AUC={auc_val:.3f})')

axes[0].plot([0,1],[0,1], '--', color='grey', linewidth=1, label='Random (0.500)')
axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
axes[0].set_ylabel('True Positive Rate (Sensitivity / Recall)')
axes[0].set_title('ROC Curves', fontweight='bold', color=C['navy'])
axes[0].legend(fontsize=8, loc='lower right')
axes[0].fill_between([0,1],[0,1], alpha=0.04, color='grey')

# Bar comparison
metrics_bar   = ['AUC-ROC', 'Gini', 'KS Statistic']
model_names   = list(results.keys())
x             = np.arange(len(metrics_bar))
width         = 0.8 / len(model_names)

for i, name in enumerate(model_names):
    vals  = [results[name][m] for m in metrics_bar]
    color = list(MODEL_COLORS.values())[i % len(MODEL_COLORS)]
    bars  = axes[1].bar(x + i*width - (len(model_names)-1)*width/2,
                        vals, width*0.9, label=name, color=color, alpha=0.85)

axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics_bar, fontsize=10)
axes[1].set_ylim(0, 1.05)
axes[1].set_title('Metric Comparison', fontweight='bold', color=C['navy'])
axes[1].legend(fontsize=8)
plt.tight_layout()
save_fig(fig, '10_roc_comparison')

# ── CHART: Confusion Matrices ──────────────────────────────────────────────────
print("  Generating confusion matrix chart...")
n_models = len(trained_models)
fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 5))
if n_models == 1: axes = [axes]
fig.suptitle('Figure 11: Confusion Matrices — All Models (Threshold = 0.50)',
             fontsize=13, fontweight='bold', color=C['navy'])
for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = (model.predict_proba(X_test)[:,1] > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Non-Def','Default'],
                yticklabels=['Non-Def','Default'],
                annot_kws={'size':10})
    ax.set_title(name, fontweight='bold', color=C['navy'], fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
save_fig(fig, '11_confusion_matrices')

# ── CHART: Precision-Recall Curves ────────────────────────────────────────────
print("  Generating precision-recall curves...")
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle('Figure 12: Precision-Recall Curves — All Models',
             fontsize=13, fontweight='bold', color=C['navy'])
for i, (name, model) in enumerate(trained_models.items()):
    y_prob   = model.predict_proba(X_test)[:,1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)
    color    = list(MODEL_COLORS.values())[i % len(MODEL_COLORS)]
    ax.plot(rec, prec, color=color, linewidth=2,
            linestyle=line_styles[i % len(line_styles)],
            label=f'{name} (PR-AUC={pr_auc:.3f})')
baseline = y_test.mean()
ax.axhline(baseline, color='grey', linestyle='--', linewidth=1,
           label=f'Baseline (No skill = {baseline:.3f})')
ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontweight='bold', color=C['navy'])
ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, '12_precision_recall_curves')


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
section(9, "FEATURE IMPORTANCE ANALYSIS")

# Use best model (highest AUC) for permutation importance
best_name  = results_df['AUC-ROC'].idxmax()
best_model = trained_models[best_name]
print(f"\n  Best model: {best_name} (AUC = {results_df.loc[best_name,'AUC-ROC']:.4f})")
print("  Computing permutation importance (may take 1–2 minutes)...")

perm = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10, random_state=42, scoring='roc_auc', n_jobs=-1)

imp_df = pd.DataFrame({
    'Feature'    : FEATURES,
    'Importance' : perm.importances_mean,
    'Std'        : perm.importances_std,
}).sort_values('Importance', ascending=False).reset_index(drop=True)

imp_df.to_csv('outputs/results/11_feature_importance.csv', index=False)
print("\n  TOP 10 FEATURES (Permutation Importance):")
print("  " + "─"*55)
for _, row in imp_df.head(10).iterrows():
    bar = '█' * max(1, int(row['Importance'] / imp_df['Importance'].max() * 25))
    print(f"  {row['Feature']:<30s} {bar} {row['Importance']:.5f}")

# Random Forest built-in importance (if RF is trained)
if 'Random Forest' in trained_models:
    rf_model = trained_models['Random Forest']
    rf_imp   = pd.DataFrame({
        'Feature'   : FEATURES,
        'RF_Importance': rf_model.feature_importances_,
    }).sort_values('RF_Importance', ascending=False)
    rf_imp.to_csv('outputs/results/12_rf_feature_importance.csv', index=False)

# Chart
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Figure 13: Feature Importance Analysis',
             fontsize=13, fontweight='bold', color=C['navy'])

top12 = imp_df.head(12).sort_values('Importance')
colors_imp = [C['red'] if imp > imp_df['Importance'].quantile(0.7) else C['blue']
              for imp in top12['Importance']]
axes[0].barh(top12['Feature'], top12['Importance'], xerr=top12['Std'],
             color=colors_imp, alpha=0.85, edgecolor='white', capsize=3)
axes[0].set_xlabel('Permutation Importance (AUC decrease)')
axes[0].set_title(f'Permutation Importance\n({best_name})',
                  fontweight='bold', color=C['navy'])
r_patch = mpatches.Patch(color=C['red'],  label='High importance')
b_patch = mpatches.Patch(color=C['blue'], label='Lower importance')
axes[0].legend(handles=[r_patch, b_patch], fontsize=9)

if 'Random Forest' in trained_models:
    rf_top = rf_imp.head(12).sort_values('RF_Importance')
    axes[1].barh(rf_top['Feature'], rf_top['RF_Importance'],
                 color=C['green'], alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('Mean Decrease in Impurity')
    axes[1].set_title('Random Forest — Built-in Importance',
                      fontweight='bold', color=C['navy'])
plt.tight_layout()
save_fig(fig, '13_feature_importance')


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — OOT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
section(10, "OUT-OF-TIME VALIDATION")

oot_results = {}
print()
for name, model in trained_models.items():
    y_prob_oot = model.predict_proba(X_oot)[:,1]
    oot_auc    = roc_auc_score(y_oot, y_prob_oot)
    oot_ks     = ks_stat(y_oot, y_prob_oot)
    test_auc   = results[name]['AUC-ROC']
    drift      = test_auc - oot_auc
    if abs(drift) < 0.02:    status = 'STABLE'
    elif abs(drift) < 0.05:  status = 'MILD DRIFT'
    else:                    status = 'HIGH DRIFT'

    oot_results[name] = {
        'Test AUC' : round(test_auc,  4),
        'OOT AUC'  : round(oot_auc,   4),
        'Drift'    : round(drift,      4),
        'OOT KS'   : round(oot_ks,     4),
        'Status'   : status,
    }
    print(f"  {name:<28s} Test={test_auc:.4f}  OOT={oot_auc:.4f}  "
          f"Drift={drift:+.4f}  {status}")

oot_df = pd.DataFrame(oot_results).T
oot_df.to_csv('outputs/results/13_oot_validation.csv')
print("\n  [SAVED] outputs/results/13_oot_validation.csv")

# OOT score distribution chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 14: Score Distribution — Test vs Out-of-Time (Best Model)',
             fontsize=13, fontweight='bold', color=C['navy'])
for ax, (label, X_set, y_set) in zip(axes, [
        ('Test Set',         X_test, y_test),
        ('Out-of-Time Set',  X_oot,  y_oot)]):
    y_prob = best_model.predict_proba(X_set)[:,1]
    ax.hist(y_prob[y_set==0], bins=50, alpha=0.6, color=C['blue'],
            label='Non-Default', density=True)
    ax.hist(y_prob[y_set==1], bins=50, alpha=0.6, color=C['red'],
            label='Default', density=True)
    auc_v = roc_auc_score(y_set, y_prob)
    ks_v  = ks_stat(y_set, y_prob)
    ax.set_title(f'{label}\nAUC={auc_v:.4f}  KS={ks_v:.4f}',
                 fontweight='bold', color=C['navy'])
    ax.set_xlabel('Predicted Default Probability')
    ax.set_ylabel('Density')
    ax.legend()
plt.tight_layout()
save_fig(fig, '14_oot_score_distribution')


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — BUSINESS SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
section(11, "PROFIT-BY-CUTOFF BUSINESS SIMULATION")

AVG_LOAN       = 250000
INT_RATE       = 0.12
LGD            = 0.65
OP_COST        = 2500
INT_INCOME     = AVG_LOAN * INT_RATE
LOSS_ON_DEF    = AVG_LOAN * LGD

print(f"\n  Assumptions:")
print(f"  Average loan amount : ₹{AVG_LOAN:,}")
print(f"  Interest income     : ₹{INT_INCOME:,} per good loan")
print(f"  Loss on default     : ₹{LOSS_ON_DEF:,} per bad loan (LGD={LGD:.0%})")
print(f"  Operating cost      : ₹{OP_COST:,} per application")

thresholds = np.linspace(0.05, 0.95, 200)
sim_results = {}

for name, model in trained_models.items():
    y_prob  = model.predict_proba(X_test)[:,1]
    profits, approval_rates = [], []
    for t in thresholds:
        reject_mask   = y_prob >= t          # predicted default → reject
        approve_mask  = ~reject_mask
        good_approved = ((approve_mask) & (y_test==0)).sum()
        bad_approved  = ((approve_mask) & (y_test==1)).sum()
        n_approved    = approve_mask.sum()
        profit = (good_approved * INT_INCOME
                - bad_approved  * LOSS_ON_DEF
                - n_approved    * OP_COST)
        profits.append(profit)
        approval_rates.append(approve_mask.mean())

    profits        = np.array(profits)
    approval_rates = np.array(approval_rates)
    opt_idx        = np.argmax(profits)
    flat_good      = (y_test==0).sum()
    flat_bad       = (y_test==1).sum()
    flat_profit    = (flat_good*INT_INCOME - flat_bad*LOSS_ON_DEF
                      - len(y_test)*OP_COST)
    improvement    = (profits[opt_idx] - flat_profit) / abs(flat_profit) * 100

    sim_results[name] = {
        'optimal_threshold'   : round(float(thresholds[opt_idx]), 3),
        'optimal_approval'    : round(float(approval_rates[opt_idx]), 4),
        'model_profit'        : round(float(profits[opt_idx]), 0),
        'baseline_profit'     : round(float(flat_profit), 0),
        'profit_improvement'  : round(float(improvement), 2),
        'thresholds'          : thresholds.tolist(),
        'profits'             : profits.tolist(),
        'approval_rates'      : approval_rates.tolist(),
    }
    print(f"\n  {name}:")
    print(f"    Optimal threshold : {thresholds[opt_idx]:.2f}")
    print(f"    Approval rate     : {approval_rates[opt_idx]:.1%}")
    print(f"    Model profit      : ₹{profits[opt_idx]:,.0f}")
    print(f"    Profit vs baseline: {improvement:+.1f}%")

# Save (without large lists)
sim_save = {k: {kk: vv for kk, vv in v.items()
                if kk not in ['thresholds','profits','approval_rates']}
            for k, v in sim_results.items()}
with open('outputs/results/14_business_simulation.json', 'w') as f:
    json.dump(sim_save, f, indent=2)

# Profit chart (best model highlighted)
best_sim = max(sim_results, key=lambda k: sim_results[k]['profit_improvement'])
print(f"\n  Best model for profit: {best_sim}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Figure 15: Profit-by-Cutoff Simulation — All Models',
             fontsize=13, fontweight='bold', color=C['navy'])

for i, (name, sim) in enumerate(sim_results.items()):
    color = list(MODEL_COLORS.values())[i % len(MODEL_COLORS)]
    lw    = 2.5 if name == best_sim else 1.2
    alpha = 1.0 if name == best_sim else 0.5
    axes[0].plot(sim['thresholds'],
                 [p/1e6 for p in sim['profits']],
                 color=color, linewidth=lw, alpha=alpha, label=name)

baseline_m = sim_results[list(sim_results.keys())[0]]['baseline_profit'] / 1e6
axes[0].axhline(baseline_m, color='grey', linestyle=':', linewidth=1.5,
                label=f'Baseline (₹{baseline_m:.2f}M)')
axes[0].set_xlabel('Decision Threshold')
axes[0].set_ylabel('Portfolio Profit (₹ Millions)')
axes[0].set_title('Profit by Approval Threshold', fontweight='bold', color=C['navy'])
axes[0].legend(fontsize=8)

for i, (name, sim) in enumerate(sim_results.items()):
    color = list(MODEL_COLORS.values())[i % len(MODEL_COLORS)]
    lw    = 2.5 if name == best_sim else 1.2
    alpha = 1.0 if name == best_sim else 0.5
    axes[1].plot(sim['thresholds'],
                 [r*100 for r in sim['approval_rates']],
                 color=color, linewidth=lw, alpha=alpha, label=name)

axes[1].set_xlabel('Decision Threshold')
axes[1].set_ylabel('Loan Approval Rate (%)')
axes[1].set_title('Approval Rate by Threshold', fontweight='bold', color=C['navy'])
axes[1].legend(fontsize=8)
plt.tight_layout()
save_fig(fig, '15_profit_by_cutoff')

# ── Risk-based pricing tiers ──────────────────────────────────────────────────
tiers = [
    {'band':'Low Risk',       'lo':0.00,'hi':0.10,'rate':'9.5–10.5%', 'policy':'Auto-approve'},
    {'band':'Moderate Risk',  'lo':0.10,'hi':0.25,'rate':'11–13%',    'policy':'Standard review'},
    {'band':'Elevated Risk',  'lo':0.25,'hi':0.45,'rate':'14–17%',    'policy':'Enhanced documentation'},
    {'band':'High Risk',      'lo':0.45,'hi':0.65,'rate':'18–22%',    'policy':'Senior approval required'},
    {'band':'Very High Risk', 'lo':0.65,'hi':1.00,'rate':'Decline',   'policy':'Decline application'},
]
best_model_obj = trained_models[best_sim]
y_prob_best    = best_model_obj.predict_proba(X_test)[:,1]
for t in tiers:
    mask        = (y_prob_best >= t['lo']) & (y_prob_best < t['hi'])
    t['count']  = int(mask.sum())
    t['pct']    = round(float(mask.mean() * 100), 1)
    t['actual_default_rate'] = round(float(y_test[mask].mean() * 100) if mask.sum()>0 else 0, 1)

tiers_df = pd.DataFrame(tiers)
tiers_df.to_csv('outputs/results/15_risk_tiers.csv', index=False)
print("\n  RISK TIER SUMMARY:")
for t in tiers:
    print(f"  {t['band']:<20s} {t['pct']:5.1f}% of applicants  "
          f"Actual default rate: {t['actual_default_rate']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — NEURAL NETWORK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section(12, "NEURAL NETWORK DEEP DIVE")

nn_names = [n for n in trained_models if 'Neural' in n]
if nn_names:
    for nn_name in nn_names:
        nn_model = trained_models[nn_name]
        y_prob_nn = nn_model.predict_proba(X_test)[:,1]

        # Decision boundary score distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Figure 16: {nn_name} — Score Analysis',
                     fontsize=13, fontweight='bold', color=C['navy'])
        axes[0].hist(y_prob_nn[y_test==0], bins=60, alpha=0.6,
                     color=C['blue'], label='Non-Default', density=True)
        axes[0].hist(y_prob_nn[y_test==1], bins=60, alpha=0.6,
                     color=C['red'], label='Default', density=True)
        auc_nn = roc_auc_score(y_test, y_prob_nn)
        axes[0].set_title(f'Score Distribution\nAUC={auc_nn:.4f}',
                          fontweight='bold', color=C['navy'])
        axes[0].set_xlabel('Predicted Default Probability')
        axes[0].set_ylabel('Density')
        axes[0].legend()

        # Calibration plot
        from sklearn.calibration import calibration_curve
        fraction_pos, mean_pred = calibration_curve(y_test, y_prob_nn, n_bins=15)
        axes[1].plot([0,1],[0,1], 'k--', linewidth=1, label='Perfect calibration')
        axes[1].plot(mean_pred, fraction_pos, 'o-',
                     color=C['purple'], linewidth=2, label=nn_name)
        axes[1].set_title('Calibration Plot\n(Reliability Diagram)',
                           fontweight='bold', color=C['navy'])
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Actual Fraction of Positives')
        axes[1].legend(fontsize=9)
        plt.tight_layout()
        save_fig(fig, f"16_neural_network_analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section(13, "FINAL SUMMARY")

best_by_auc   = results_df['AUC-ROC'].idxmax()
best_by_f1    = results_df['F1 (Default)'].idxmax()
best_by_ks    = results_df['KS Statistic'].idxmax()
best_by_profit= max(sim_results, key=lambda k: sim_results[k]['profit_improvement'])

summary = {
    'dataset'                    : 'Give Me Some Credit (Kaggle)',
    'total_records'              : int(len(df)),
    'default_rate'               : float(df['default'].mean()),
    'imbalance_ratio'            : float((df['default']==0).sum()/(df['default']==1).sum()),
    'features_used'              : len(FEATURES),
    'models_trained'             : list(trained_models.keys()),
    'best_model_auc'             : best_by_auc,
    'best_auc_value'             : float(results_df.loc[best_by_auc,'AUC-ROC']),
    'best_model_f1'              : best_by_f1,
    'best_f1_value'              : float(results_df.loc[best_by_f1,'F1 (Default)']),
    'best_model_ks'              : best_by_ks,
    'best_ks_value'              : float(results_df.loc[best_by_ks,'KS Statistic']),
    'best_model_profit'          : best_by_profit,
    'best_profit_improvement_pct': float(sim_results[best_by_profit]['profit_improvement']),
    'optimal_threshold'          : float(sim_results[best_by_profit]['optimal_threshold']),
    'optimal_approval_rate'      : float(sim_results[best_by_profit]['optimal_approval']),
    'oot_all_stable'             : all(v['Status']=='STABLE' for v in oot_results.values()),
    'top_feature'                : str(imp_df.iloc[0]['Feature']),
    'top_feature_importance'     : float(imp_df.iloc[0]['Importance']),
    'all_model_results'          : {k: {kk: float(vv) if isinstance(vv, (int,float,np.floating)) else vv
                                        for kk, vv in v.items()}
                                    for k, v in results.items()},
    'oot_results'                : {k: {kk: float(vv) if isinstance(vv, (int,float,np.floating)) else str(vv)
                                        for kk, vv in v.items()}
                                    for k, v in oot_results.items()},
}

with open('outputs/results/00_MASTER_SUMMARY.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  ANALYSIS COMPLETE — RESULTS SUMMARY
  ╠══════════════════════════════════════════════════════════════╣
  ║  Dataset        : {summary['dataset']:<41s}║
  ║  Records        : {summary['total_records']:,} borrowers{'':<32s}║
  ║  Default rate   : {summary['default_rate']:.2%}{'':<41s}║
  ║  Models trained : {len(trained_models)}{'':<43s}║
  ╠══════════════════════════════════════════════════════════════╣
  ║  BEST AUC-ROC   : {best_by_auc:<25s}  {summary['best_auc_value']:.4f}  ║
  ║  BEST F1-Score  : {best_by_f1:<25s}  {summary['best_f1_value']:.4f}  ║
  ║  BEST KS-Stat   : {best_by_ks:<25s}  {summary['best_ks_value']:.4f}  ║
  ║  BEST Profit +  : {best_by_profit:<25s}  +{summary['best_profit_improvement_pct']:.1f}%  ║
  ║  OOT Stable     : {'All models ✅' if summary['oot_all_stable'] else 'Some drift ⚠️':<44s}║
  ║  Top Feature    : {summary['top_feature']:<44s}║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Upload the entire outputs/ folder to Claude for the        ║
  ║  complete final report with your real results embedded.     ║
  ╚══════════════════════════════════════════════════════════════╝
""")

print("  FILES TO UPLOAD TO CLAUDE:")
print("  ─"*35)
for fname in sorted(os.listdir('outputs/results/')):
    print(f"  outputs/results/{fname}")
print()
for fname in sorted(os.listdir('outputs/charts/')):
    print(f"  outputs/charts/{fname}")