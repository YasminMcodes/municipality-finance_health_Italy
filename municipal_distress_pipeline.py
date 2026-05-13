# ============================================================
# COMPLETE PIPELINE SCRIPT — WITH HYPERPARAMETER OPTIMIZATION
# Municipal Fiscal Distress — Italy 2016-2023
# From data loading to figure generation
#
# Changes vs. previous version:
#   • Step 6.5  — Randomized Search hyperparameter optimization
#                (thresholds kept at original 0.50 / 0.40 / 0.15)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import shap
from scipy.stats import loguniform, uniform
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    classification_report, average_precision_score, brier_score_loss,
    roc_curve, auc, precision_recall_curve,
)
from xgboost import XGBClassifier

# ── Create output folders ─────────────────────────────────────
os.makedirs('figures', exist_ok=True)
os.makedirs('figures/waterfalls', exist_ok=True)
print("✓ Output folders created")

# ── Shared style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COLORS = {
    'bankrupt':     '#D65F5F',
    'non_bankrupt': '#4878CF',
    'xgb':          '#D65F5F',
    'rf':           '#6ACC65',
    'lr':           '#4878CF',
}

RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1 — Loading Data")
print("="*60)

df = pd.read_csv('final_municipality_dataset_percap_with_indicators.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════
# STEP 2 — CLEAN
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2 — Cleaning")
print("="*60)

non_financial = ['Municipality_ID', 'Municipality', 'Province',
                 'Year', 'Region_Group', 'Population_2019', 'Bankrupt']
financial_cols = [c for c in df.columns if c not in non_financial]

all_missing_mask = df[financial_cols].isna().all(axis=1)
df_clean = df.loc[~all_missing_mask].copy()

print(f"Original rows  : {len(df)}")
print(f"Dropped rows   : {all_missing_mask.sum()}")
print(f"Rows kept      : {len(df_clean)}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — FEATURES & TARGET
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3 — Defining Features and Target")
print("="*60)

target   = 'Bankrupt'
id_like  = ['Municipality_ID', 'Municipality', 'Province']

# Drop Year from X completely
X = df_clean.drop(columns=[target] + id_like + ['Year'])
y = df_clean[target]

# Region_Group as categorical only (Year removed)
categorical_cols = ['Region_Group']
numeric_cols     = [c for c in X.columns if c not in categorical_cols]

print(f"Features       : {X.shape[1]}")
print(f"Target         : {target}")
print(f"Categorical    : {categorical_cols}")
print(f"Numeric        : {len(numeric_cols)} columns")

# ══════════════════════════════════════════════════════════════
# STEP 4 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4 — Train/Test Split")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

def show_balance(name, y):
    vc = y.value_counts(normalize=True)
    print(f"{name} | size={len(y)} | "
          f"non-bankrupt={vc.get(0,0):.3f} | "
          f"bankrupt={vc.get(1,0):.3f}")

show_balance("TRAIN", y_train)
show_balance("TEST ", y_test)

# ══════════════════════════════════════════════════════════════
# STEP 5 — PREPROCESSORS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5 — Building Preprocessors")
print("="*60)

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocess_lr = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', num_imputer),
                      ('scaler', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('imp', cat_imputer),
                      ('enc', cat_encoder)]),          categorical_cols),
], remainder='drop')

preprocess_tree = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', num_imputer)]), numeric_cols),
    ('cat', Pipeline([('imp', cat_imputer),
                      ('enc', cat_encoder)]), categorical_cols),
], remainder='drop')

print("✓ Preprocessors built")

# ══════════════════════════════════════════════════════════════
# STEP 6 — MODELS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6 — Defining Models")
print("="*60)

log_reg = LogisticRegression(
    penalty='l2', C=1.0, solver='lbfgs',
    max_iter=500, random_state=RANDOM_STATE
)

rf = RandomForestClassifier(
    n_estimators=400, max_features='sqrt',
    n_jobs=-1, random_state=RANDOM_STATE
)

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=400, max_depth=6,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg / pos,
    eval_metric='logloss',
    n_jobs=-1, random_state=RANDOM_STATE
)

print(f"scale_pos_weight: {neg/pos:.3f}")

# ── Pipelines ─────────────────────────────────────────────────
pipe_lr  = Pipeline([('prep', preprocess_lr),  ('clf', log_reg)])
pipe_rf  = Pipeline([('prep', preprocess_tree), ('clf', rf)])
pipe_xgb = Pipeline([('prep', preprocess_tree), ('clf', xgb)])
print("✓ Pipelines built")

# ══════════════════════════════════════════════════════════════
# STEP 6.5 — HYPERPARAMETER OPTIMIZATION (Randomized Search)
# Scored by PR-AUC, 5-fold Stratified CV.
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6.5 — Hyperparameter Optimization (Randomized Search)")
print("="*60)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True,
                           random_state=RANDOM_STATE)
SEARCH_SCORING = 'average_precision'

# ── Search space: Logistic Regression ────────────────────────
param_dist_lr = {
    'clf__C':            loguniform(1e-3, 1e2),
    'clf__penalty':      ['l1', 'l2'],
    'clf__class_weight': [None, 'balanced'],
    'clf__solver':       ['liblinear'],
}

# ── Search space: Random Forest ──────────────────────────────
param_dist_rf = {
    'clf__n_estimators':      [200, 400, 600, 800],
    'clf__max_depth':         [None, 5, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf':  [1, 2, 4],
    'clf__max_features':      ['sqrt', 'log2', 0.3, 0.5],
    'clf__class_weight':      [None, 'balanced', 'balanced_subsample'],
}

# ── Search space: XGBoost ────────────────────────────────────
# scale_pos_weight kept fixed at neg/pos (set in Step 6).
param_dist_xgb = {
    'clf__n_estimators':     [200, 400, 600, 800, 1000],
    'clf__max_depth':        [3, 4, 5, 6, 8, 10],
    'clf__learning_rate':    loguniform(1e-2, 3e-1),
    'clf__subsample':        uniform(0.6, 0.4),   # 0.6 → 1.0
    'clf__colsample_bytree': uniform(0.6, 0.4),   # 0.6 → 1.0
    'clf__min_child_weight': [1, 3, 5, 7],
    'clf__gamma':            [0, 0.1, 0.3, 0.5],
    'clf__reg_alpha':        loguniform(1e-3, 1e1),
    'clf__reg_lambda':       loguniform(1e-3, 1e1),
}

def run_search(name, pipe, param_dist, n_iter):
    print(f"\n  Tuning {name}  "
          f"({n_iter} iterations × 5 folds = {n_iter*5} fits)")
    print("  " + "-"*56)
    search = RandomizedSearchCV(
        estimator           = pipe,
        param_distributions = param_dist,
        n_iter              = n_iter,
        scoring             = SEARCH_SCORING,
        cv                  = inner_cv,
        n_jobs              = -1,
        random_state        = RANDOM_STATE,
        refit               = True,
        verbose             = 0,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"  Best params:")
    for k, v in search.best_params_.items():
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<26} = {v_str}")
    return search.best_estimator_, search.best_score_

# Reassign pipelines to the tuned versions
pipe_lr,  best_lr_score  = run_search(
    "Logistic Regression", pipe_lr,  param_dist_lr,  n_iter=20)
pipe_rf,  best_rf_score  = run_search(
    "Random Forest",       pipe_rf,  param_dist_rf,  n_iter=30)
pipe_xgb, best_xgb_score = run_search(
    "XGBoost",             pipe_xgb, param_dist_xgb, n_iter=50)

print("\n" + "="*60)
print("  TUNED MODELS — Best CV PR-AUC Summary")
print("="*60)
print(f"  {'Model':<22} {'Best CV PR-AUC':>16}")
print(f"  {'-'*40}")
print(f"  {'Logistic Regression':<22} {best_lr_score:>16.4f}")
print(f"  {'Random Forest':<22} {best_rf_score:>16.4f}")
print(f"  {'XGBoost':<22} {best_xgb_score:>16.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 7 — CROSS VALIDATION (now on tuned pipelines)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 7 — Cross Validation")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    'accuracy':          'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision':         'precision',
    'recall':            'recall',
    'f1':                'f1',
    'average_precision': 'average_precision',
    'roc_auc':           'roc_auc',
}

def evaluate_model(name, pipe):
    print(f"\n{'='*60}")
    print(f"  {name} — 5-fold CV on TRAIN")
    print(f"{'='*60}")
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=scoring, n_jobs=-1,
                            return_train_score=False)
    metrics = [
        ('accuracy',          'Accuracy'),
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('precision',         'Precision'),
        ('recall',            'Recall'),
        ('f1',                'F1-score'),
        ('average_precision', 'PR-AUC'),
        ('roc_auc',           'ROC-AUC'),
    ]
    for key, label in metrics:
        mean = cv_res[f'test_{key}'].mean()
        std  = cv_res[f'test_{key}'].std()
        print(f"  {label:<22}: {mean:.3f} ± {std:.3f}")
    return pipe.fit(X_train, y_train)

model_lr  = evaluate_model("Logistic Regression", pipe_lr)
model_rf  = evaluate_model("Random Forest",       pipe_rf)
model_xgb = evaluate_model("XGBoost",             pipe_xgb)

# ══════════════════════════════════════════════════════════════
# STEP 8 — HELD-OUT TEST EVALUATION (uses tuned thresholds)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8 — Held-Out Test Evaluation")
print("="*60)

def evaluate_on_test(name, model, X_test, y_test, threshold=0.50):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    acc   = accuracy_score(y_test, preds)
    bacc  = balanced_accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, zero_division=0)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)
    prauc = average_precision_score(y_test, proba)
    roc   = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(
        y_test, preds, labels=[0, 1]).ravel()

    print(f"\n{'='*60}")
    print(f"  {name} — Held-out TEST (threshold={threshold:.2f})")
    print(f"{'='*60}")
    print(f"  Accuracy          : {acc:.3f}")
    print(f"  Balanced Accuracy : {bacc:.3f}")
    print(f"  Precision         : {prec:.3f}")
    print(f"  Recall            : {rec:.3f}")
    print(f"  F1-score          : {f1:.3f}")
    print(f"  PR-AUC            : {prauc:.3f}  <- primary metric")
    print(f"  ROC-AUC           : {roc:.3f}")
    print(f"  Brier Score       : {brier:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"               Pred 0   Pred 1")
    print(f"  True 0 (N)   {tn:6d}   {fp:6d}")
    print(f"  True 1 (B)   {fn:6d}   {tp:6d}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, preds,
        target_names=['Non-Bankrupt (0)', 'Bankrupt (1)'],
        digits=3))

    return {'accuracy': acc, 'balanced_accuracy': bacc,
            'precision': prec, 'recall': rec, 'f1': f1,
            'pr_auc': prauc, 'roc_auc': roc, 'brier': brier,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

res_lr  = evaluate_on_test("Logistic Regression", model_lr,
                            X_test, y_test, threshold=0.50)
res_rf  = evaluate_on_test("Random Forest",       model_rf,
                            X_test, y_test, threshold=0.40)
res_xgb = evaluate_on_test("XGBoost",             model_xgb,
                            X_test, y_test, threshold=0.15)

print(f"\n{'='*60}")
print("  FINAL COMPARISON — Held-out TEST")
print(f"{'='*60}")
print(f"  {'Model':<22} {'PR-AUC':>8} {'ROC-AUC':>9} "
      f"{'F1':>7} {'Recall':>8} {'Brier':>7}")
print(f"  {'-'*57}")
for name, res in [
    ("Logistic Regression", res_lr),
    ("Random Forest",       res_rf),
    ("XGBoost",             res_xgb),
]:
    print(f"  {name:<22} "
          f"{res['pr_auc']:>8.3f} "
          f"{res['roc_auc']:>9.3f} "
          f"{res['f1']:>7.3f} "
          f"{res['recall']:>8.3f} "
          f"{res['brier']:>7.3f}")

# ══════════════════════════════════════════════════════════════
# STEP 9 — CLEAN LABEL HELPER
# ══════════════════════════════════════════════════════════════
def clean_label(name):
    name = name.replace('num__', '').replace('cat__', '')
    name = name.replace('_pc', '')
    replacements = {
        'Current Tax, Contribution & Equalization Revenues':
            'Tax & Equalization Revenues',
        'Revenues from Financial Asset Reduction':
            'Asset Liquidation Revenues',
        'Treasurer/Cashier Advances':
            'Cashier Advances',
        'Third-Party & Giro Revenues':
            'Third-Party Revenues',
        'Education & Right to Study':
            'Education Spending',
        'Social Rights, Policies & Family':
            'Social Rights & Welfare',
        'Sustainable Development & Environmental Protection':
            'Sustainable Development',
        'Agriculture, Agri-food Policies & Fisheries':
            'Agriculture Policies',
        'Economic Development & Competitiveness':
            'Economic Development',
        'Energy & Diversification of Sources':
            'Energy & Diversification',
        'Transport & Right to Mobility':
            'Transport & Mobility',
        'Employment & Vocational Training':
            'Employment & Training',
        'Youth Policies, Sports & Leisure':
            'Youth & Sports',
        'Population aged 25-64 with an educational qualification '
        'no higher than lower secondary education':
            'Low Education Rate (25-64)',
    }
    for long, short in replacements.items():
        name = name.replace(long, short)
    return name.strip()

# ══════════════════════════════════════════════════════════════
# STEP 10 — FIGURE GENERATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 10 — Generating Figures")
print("="*60)

# ── Shared data for figures ───────────────────────────────────
feature_names_out = model_xgb.named_steps['prep'].get_feature_names_out()
clean_names       = [clean_label(n) for n in feature_names_out]
X_test_prep       = model_xgb.named_steps['prep'].transform(X_test)
xgb_clf           = model_xgb.named_steps['clf']

proba_lr  = model_lr.predict_proba(X_test)[:, 1]
proba_rf  = model_rf.predict_proba(X_test)[:, 1]
proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

# ── Figure 1: Class Balance ───────────────────────────────────
print("\nGenerating Figure 1: Class Balance...")

bankrupt_n     = int(y.sum())
non_bankrupt_n = len(y) - bankrupt_n

fig, ax = plt.subplots(figsize=(6, 5))
wedges, texts, autotexts = ax.pie(
    [bankrupt_n, non_bankrupt_n],
    explode=(0.05, 0),
    labels=[f'Bankrupt\n(n={bankrupt_n})',
            f'Non-Bankrupt\n(n={non_bankrupt_n})'],
    colors=[COLORS['bankrupt'], COLORS['non_bankrupt']],
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
)
for t in texts:
    t.set_fontsize(11); t.set_fontweight('bold')
for at in autotexts:
    at.set_fontsize(12); at.set_fontweight('bold')
    at.set_color('white')
ax.text(0, 0, f'N = {len(y)}\nMunicipalities',
        ha='center', va='center',
        fontsize=11, fontweight='bold', color='#333333')
ax.set_title('Class Distribution in Sample\n(N = 462 Municipalities)',
             fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/class_balance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 1 saved: figures/class_balance.png")

# ── Figure 2: Missing Data ────────────────────────────────────
print("Generating Figure 2: Missing Data...")

df_financial  = df[financial_cols]
missing_all   = (df_financial.isnull().sum()
                 / len(df_financial) * 100)
missing_all   = missing_all[missing_all > 0].sort_values(ascending=False)
top_missing   = missing_all.head(10)

df_b  = df[df['Bankrupt'] == 1][financial_cols]
df_nb = df[df['Bankrupt'] == 0][financial_cols]
missing_b  = (df_b[top_missing.index].isnull().sum()
              / len(df_b) * 100)
missing_nb = (df_nb[top_missing.index].isnull().sum()
              / len(df_nb) * 100)

def clean_col_name(col):
    return col.replace('_pc', ' (p.c.)').replace('_', ' ').title()

labels_m = [clean_col_name(c) for c in top_missing.index]
x_m      = np.arange(len(labels_m))
width_m  = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars_b  = ax.bar(x_m - width_m/2, missing_b.values, width_m,
                 label='Bankrupt', color=COLORS['bankrupt'], alpha=0.85)
bars_nb = ax.bar(x_m + width_m/2, missing_nb.values, width_m,
                 label='Non-Bankrupt', color=COLORS['non_bankrupt'],
                 alpha=0.85)
for bar in bars_b:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h:.0f}%', ha='center', va='bottom',
                fontsize=9, color=COLORS['bankrupt'], fontweight='bold')
for bar in bars_nb:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h:.0f}%', ha='center', va='bottom',
                fontsize=9, color=COLORS['non_bankrupt'], fontweight='bold')
ax.set_xlabel('Financial Indicator', fontsize=12, labelpad=10)
ax.set_ylabel('Missing Values (%)', fontsize=12)
ax.set_title('Missing Financial Data by Bankruptcy Status\n'
             'Top 10 Most Incomplete Indicators',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x_m)
ax.set_xticklabels(labels_m, rotation=35, ha='right', fontsize=9)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(fontsize=11, framealpha=0.9)
ax.set_ylim(0, max(missing_b.max(), missing_nb.max()) + 12)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/missing_data_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved: figures/missing_data_analysis.png")

# ── Figure 3: Spending Comparison ────────────────────────────
print("Generating Figure 3: Spending Comparison...")

edu_col    = [c for c in df.columns
              if 'Education' in c or 'Right to Study' in c][0]
social_col = [c for c in df.columns
              if 'Social Rights' in c or 'Policies & Family' in c][0]

spending_vars = [edu_col, social_col]
means_b  = df[df['Bankrupt'] == 1][spending_vars].mean()
means_nb = df[df['Bankrupt'] == 0][spending_vars].mean()

labels_s = ['Education\nSpending (p.c.)',
            'Social Rights\nSpending (p.c.)']
x_s      = np.arange(len(labels_s))
width_s  = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars_b  = ax.bar(x_s - width_s/2, means_b.values, width_s,
                 label='Bankrupt', color=COLORS['bankrupt'], alpha=0.85)
bars_nb = ax.bar(x_s + width_s/2, means_nb.values, width_s,
                 label='Non-Bankrupt', color=COLORS['non_bankrupt'],
                 alpha=0.85)
for bar in bars_b:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f'\u20ac{h:.0f}', ha='center', va='bottom',
            fontsize=9, color=COLORS['bankrupt'], fontweight='bold')
for bar in bars_nb:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f'\u20ac{h:.0f}', ha='center', va='bottom',
            fontsize=9, color=COLORS['non_bankrupt'], fontweight='bold')
ax.set_ylabel('Mean Per Capita Spending (\u20ac)', fontsize=11)
ax.set_title('Spending Patterns: Bankrupt vs Non-Bankrupt\n'
             'Municipalities', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x_s)
ax.set_xticklabels(labels_s, fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/spending_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved: figures/spending_comparison.png")

# ── Figure 4: ROC Curve ───────────────────────────────────────
print("Generating Figure 4: ROC Curve...")

fpr_lr,  tpr_lr,  _ = roc_curve(y_test, proba_lr)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)
auc_lr  = auc(fpr_lr,  tpr_lr)
auc_rf  = auc(fpr_rf,  tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_lr,  tpr_lr,  color=COLORS['lr'],  lw=2,
        label=f'Logistic Regression  (AUC = {auc_lr:.3f})')
ax.plot(fpr_rf,  tpr_rf,  color=COLORS['rf'],  lw=2,
        label=f'Random Forest        (AUC = {auc_rf:.3f})')
ax.plot(fpr_xgb, tpr_xgb, color=COLORS['xgb'], lw=2,
        label=f'XGBoost              (AUC = {auc_xgb:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1,
        label='Random Classifier (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate',  fontsize=12)
ax.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/final_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4 saved: figures/final_roc_curve.png")

# ── Figure 5: PR Curve ────────────────────────────────────────
print("Generating Figure 5: PR Curve...")

prec_lr,  rec_lr,  _ = precision_recall_curve(y_test, proba_lr)
prec_rf,  rec_rf,  _ = precision_recall_curve(y_test, proba_rf)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, proba_xgb)
prauc_lr  = average_precision_score(y_test, proba_lr)
prauc_rf  = average_precision_score(y_test, proba_rf)
prauc_xgb = average_precision_score(y_test, proba_xgb)
baseline  = y_test.mean()

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_lr,  prec_lr,  color=COLORS['lr'],  lw=2,
        label=f'Logistic Regression  (PR-AUC = {prauc_lr:.3f})')
ax.plot(rec_rf,  prec_rf,  color=COLORS['rf'],  lw=2,
        label=f'Random Forest        (PR-AUC = {prauc_rf:.3f})')
ax.plot(rec_xgb, prec_xgb, color=COLORS['xgb'], lw=2,
        label=f'XGBoost              (PR-AUC = {prauc_xgb:.3f})')
ax.axhline(y=baseline, color='k', linestyle='--', lw=1,
           label=f'Random Classifier (PR-AUC = {baseline:.3f})')
ax.set_xlabel('Recall',    fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve Comparison',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/final_pr_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 5 saved: figures/final_pr_curve.png")

# ── Figure 6: Gain Importance ─────────────────────────────────
print("Generating Figure 6: Gain Importance...")

feature_map = {f'f{i}': name
               for i, name in enumerate(feature_names_out)}
gain_scores = xgb_clf.get_booster().get_score(importance_type='gain')
gain_named  = {feature_map.get(k, k): v
               for k, v in gain_scores.items()}
gain_df = pd.DataFrame({
    'feature': list(gain_named.keys()),
    'gain':    list(gain_named.values())
}).sort_values('gain', ascending=False).head(10)
gain_df['label'] = gain_df['feature'].apply(clean_label)

colors_gain = ['#D65F5F' if i == 0 else '#4878CF'
               for i in range(len(gain_df))]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(gain_df['label'][::-1], gain_df['gain'][::-1],
               color=colors_gain[::-1], alpha=0.85,
               edgecolor='white', linewidth=0.8)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.02, bar.get_y() + bar.get_height()/2,
            f'{w:.2f}', va='center', ha='left',
            fontsize=9, color='#333333')
ax.set_xlabel('Gain Score', fontsize=12)
ax.set_title('XGBoost \u2014 Top 10 Structural Drivers of Fiscal Distress\n'
             '(Gain: contribution to reducing prediction uncertainty)',
             fontsize=12, fontweight='bold', pad=15)
top_patch  = mpatches.Patch(color='#D65F5F', alpha=0.85,
                             label='Top predictor')
rest_patch = mpatches.Patch(color='#4878CF', alpha=0.85,
                             label='Other top predictors')
ax.legend(handles=[top_patch, rest_patch], fontsize=10, framealpha=0.9)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/gain_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 6 saved: figures/gain_importance.png")

# ── Figure 7: SHAP Summary ────────────────────────────────────
print("Generating Figure 7: SHAP Summary...")

explainer   = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test_prep)
shap_exp    = explainer(X_test_prep)

fig = plt.figure(figsize=(14, 8))
shap.summary_plot(
    shap_values, X_test_prep,
    feature_names=clean_names,
    max_display=12, show=False,
    plot_size=None, color_bar=True,
    plot_type='dot', alpha=0.8,
)
ax = plt.gca()
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
xlim = ax.get_xlim()
ax.text(xlim[1] * 0.6, -0.8,
        '\u2192 Increases bankruptcy risk',
        fontsize=10, color='#D65F5F', fontweight='bold', ha='center')
ax.text(xlim[0] * 0.6, -0.8,
        'Reduces bankruptcy risk \u2190',
        fontsize=10, color='#4878CF', fontweight='bold', ha='center')
plt.title('XGBoost SHAP Summary \u2014 Direction & Magnitude of Each Driver\n'
          'Red = high feature value   |   Blue = low feature value',
          fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 7 saved: figures/shap_summary.png")

# ── Figure 8 & 9: Waterfall Plots ────────────────────────────
print("Generating Figure 8 & 9: Waterfall Plots...")

preds_xgb  = (proba_xgb >= 0.15).astype(int)
test_munis = df_clean.iloc[X_test.index]['Municipality'].values

idx_pignataro = np.where(test_munis == 'PIGNATARO INTERAMNA')[0][0]
idx_carinola  = np.where(test_munis == 'CARINOLA')[0][0]

def make_named_exp(idx):
    return shap.Explanation(
        values        = shap_exp[idx].values,
        base_values   = shap_exp[idx].base_values,
        data          = shap_exp[idx].data,
        feature_names = clean_names
    )

exp_pignataro = make_named_exp(idx_pignataro)
exp_carinola  = make_named_exp(idx_carinola)

# Pignataro
fig = plt.figure(figsize=(12, 7))
shap.plots.waterfall(exp_pignataro, max_display=12, show=False)
plt.title(
    'Panel A \u2014 PIGNATARO INTERAMNA (FR, 2018)\n'
    'Education Spending: \u20ac129 p.c. \u2192 '
    'Partial protection (Blue bar)\n'
    f'Predicted bankruptcy probability: '
    f'{proba_xgb[idx_pignataro]:.3f}',
    fontsize=11, fontweight='bold', pad=12
)
plt.tight_layout()
plt.savefig('figures/waterfalls/waterfall_pignataro.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 8 saved: figures/waterfalls/waterfall_pignataro.png")

# Carinola
fig = plt.figure(figsize=(12, 7))
shap.plots.waterfall(exp_carinola, max_display=12, show=False)
plt.title(
    'Panel B \u2014 CARINOLA (CE, 2022)\n'
    'Education Spending: \u20ac27 p.c. \u2192 '
    'Amplifies risk (Red bar)\n'
    f'Predicted bankruptcy probability: '
    f'{proba_xgb[idx_carinola]:.3f}',
    fontsize=11, fontweight='bold', pad=12
)
plt.tight_layout()
plt.savefig('figures/waterfalls/waterfall_carinola.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 9 saved: figures/waterfalls/waterfall_carinola.png")

# ══════════════════════════════════════════════════════════════
# STEP 11 — ANONYMISED DUAL WATERFALL COMPARISON
# ──────────────────────────────────────────────────────────────
# Generates a side-by-side SHAP waterfall plot for one
# distressed municipality and one healthy peer, with all
# identifying information (name, province) suppressed.
# Macro-region and population are kept because they are part
# of the comparative argument, not municipal identifiers.
#
# Append this block to the end of your existing pipeline.
# Requires the following objects already in scope:
#   df_clean, X, model_xgb, explainer, clean_names, shap, plt
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 11 — Anonymised Dual Waterfall Comparison")
print("="*60)

import matplotlib.image as mpimg

# ── Configuration ────────────────────────────────────────────
# Real names are used only to look up the correct rows in
# df_clean. They never appear in the rendered figure.
DISTRESSED = {
    'name':  'CARINOLA',
    'year':  2022,
    'label': 'Municipality A',
}
HEALTHY = {
    'name':  'MESORACA',
    'year':  2021,
    'label': 'Municipality B',
}

# ── Helper: compute SHAP for any row in df_clean ─────────────
def shap_for_row(df_row_index):
    """SHAP values + predicted probability for one df_clean row.
    Works whether or not the row is in the held-out test set."""
    row_X = df_clean.loc[[df_row_index]].drop(
        columns=['Bankrupt', 'Municipality_ID', 'Municipality',
                 'Province', 'Year'], errors='ignore'
    )
    row_X    = row_X[X.columns]
    row_prep = model_xgb.named_steps['prep'].transform(row_X)
    exp_obj  = explainer(row_prep)
    named    = shap.Explanation(
        values        = exp_obj[0].values,
        base_values   = exp_obj[0].base_values,
        data          = exp_obj[0].data,
        feature_names = clean_names,
    )
    prob = model_xgb.predict_proba(row_X)[0, 1]
    return named, prob

def find_row(name, year):
    match = df_clean[
        (df_clean['Municipality'].str.upper() == name.upper()) &
        (df_clean['Year'] == year)
    ]
    if match.empty:
        raise ValueError(f"Row not found: {name} ({year})")
    return match.index[0], match.iloc[0]

# ── Locate rows and compute SHAP values ──────────────────────
idx_A, info_A = find_row(DISTRESSED['name'], DISTRESSED['year'])
idx_B, info_B = find_row(HEALTHY['name'],    HEALTHY['year'])

exp_A, prob_A = shap_for_row(idx_A)
exp_B, prob_B = shap_for_row(idx_B)

print(f"  {DISTRESSED['label']}: "
      f"pop = {int(info_A['Population']):,}, "
      f"region = {info_A['Region_Group']}, "
      f"P(distress) = {prob_A:.3f}")
print(f"  {HEALTHY['label']}: "
      f"pop = {int(info_B['Population']):,}, "
      f"region = {info_B['Region_Group']}, "
      f"P(distress) = {prob_B:.3f}")

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Generate the two waterfall plots (anonymised titles) ─────
def render_waterfall(exp, info, label, prob, out_path):
    fig = plt.figure(figsize=(12, 7))
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(
        f'{label}\n'
        f'Population: {int(info["Population"]):,} | '
        f'Region: {info["Region_Group"]}\n'
        f'Predicted distress probability: {prob:.3f}',
        fontsize=11, fontweight='bold', pad=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

path_A = 'figures/waterfalls/waterfall_municipality_A.png'
path_B = 'figures/waterfalls/waterfall_municipality_B.png'

render_waterfall(exp_A, info_A, DISTRESSED['label'], prob_A, path_A)
render_waterfall(exp_B, info_B, HEALTHY['label'],    prob_B, path_B)
print(f"  ✓ Saved: {path_A}")
print(f"  ✓ Saved: {path_B}")

# ── Combine into side-by-side comparison ─────────────────────
img_A = mpimg.imread(path_A)
img_B = mpimg.imread(path_B)

fig, axes = plt.subplots(1, 2, figsize=(22, 9))

axes[0].imshow(img_A)
axes[0].axis('off')
axes[0].set_title(
    f'Panel A — {DISTRESSED["label"]} ({DISTRESSED["year"]})\n'
    f'Distressed | P(distress) = {prob_A:.3f}',
    fontsize=13, fontweight='bold', pad=10, color='#D65F5F',
)

axes[1].imshow(img_B)
axes[1].axis('off')
axes[1].set_title(
    f'Panel B — {HEALTHY["label"]} ({HEALTHY["year"]})\n'
    f'Healthy Peer | P(distress) = {prob_B:.3f}',
    fontsize=13, fontweight='bold', pad=10, color='#4878CF',
)

# Round populations to nearest 100 for the suptitle range
pop_A_round = int(round(info_A['Population'] / 100.0)) * 100
pop_B_round = int(round(info_B['Population'] / 100.0)) * 100

fig.suptitle(
    f'SHAP Waterfall Comparison: Distressed vs Healthy Municipality\n'
    f'Same Region ({info_A["Region_Group"]}), '
    f'Similar Population (~{pop_A_round:,} vs ~{pop_B_round:,})',
    fontsize=14, fontweight='bold', y=1.01,
)

plt.tight_layout()
out_combined = 'figures/waterfall_comparison_anonymous.png'
plt.savefig(out_combined, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_combined}")

print("\nStep 11 complete. Anonymised figure ready for inclusion.")

# ══════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print("\nFiles saved:")
all_files = [
    "figures/class_balance.png",
    "figures/missing_data_analysis.png",
    "figures/spending_comparison.png",
    "figures/final_roc_curve.png",
    "figures/final_pr_curve.png",
    "figures/gain_importance.png",
    "figures/shap_summary.png",
    "figures/waterfalls/waterfall_pignataro.png",
    "figures/waterfalls/waterfall_carinola.png",
    "figures/waterfalls/waterfall_municipality_A.png",
    "figures/waterfalls/waterfall_municipality_B.png",
    "figures/waterfall_comparison_anonymous.png",
]
for f in all_files:
    print(f"  ✓ {f}")
