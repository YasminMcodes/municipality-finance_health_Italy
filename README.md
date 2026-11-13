# 🏛️ Municipal Fiscal Distress Prediction — Italy 2016–2023

A statistical analysis of social, economic, and financial indicators associated with municipal fiscal fragility in Italy, using an explainable machine learning framework.

This repository contains the analysis pipeline, dataset, visualizations, and numerical results supporting the research.

---

## 🎯 Research Objective

This project investigates the structural determinants of municipal fiscal fragility in Italy during the 2016–2023 period.

Beyond predictive performance, the study uses explainable machine learning to examine the financial, economic, and socio-demographic characteristics associated with municipal distress.

The objective is to contribute evidence that may support fiscal monitoring and the identification of municipalities exhibiting characteristics associated with financial vulnerability.

### Research Hypotheses

Three hypotheses guide the analysis:

- **H1:** Administrative deficits and cashier advances are dominant indicators of fiscal distress, independently of debt levels.
- **H2:** Education spending acts as a structural protective factor, while welfare expenditure functions as a contextual risk segmenter.
- **H3:** Socio-demographic fragility, including low educational attainment and population decline, amplifies distress risk independently of financial indicators.

These hypotheses are investigated through predictive modeling, feature attribution, and robustness analyses. Model explanations indicate associations and predictive contributions rather than establishing causal effects.

---

## 📊 Data Sources

| Source | Content |
|---|---|
| [Open Bilanci](https://openbilanci.it/) | Municipal budget and financial indicators |
| [Banca Dati Ca' Foscari](https://www.bancadaticomunicf.eu/public/elen_comu.php) | Municipal fiscal distress procedure records |
| [ISTAT](https://www.istat.it/) | Socio-demographic indicators |

### Sample Design

The dataset follows a matched case-control design, covering municipalities associated with fiscal distress procedures and a comparison group of non-distressed municipalities.

The analytical dataset includes **462 municipalities**:

- **195 distressed municipalities**
- **267 non-distressed municipalities**
- Observation period: **2016–2023**

Because the sample deliberately includes a substantial proportion of distressed municipalities, its class distribution should not be interpreted as the national prevalence of municipal fiscal distress.

---

## 🧠 Machine Learning Methodology

### Models

Three supervised classification algorithms are evaluated:

1. **Logistic Regression** — interpretable linear baseline.
2. **Random Forest** — ensemble classification using decision trees.
3. **XGBoost** — gradient-boosted decision-tree classification.

### Model Training and Evaluation

The modeling workflow includes:

- Stratified training and held-out test split (70/30).
- Data preprocessing and missing-value handling.
- Hyperparameter optimization using `RandomizedSearchCV`.
- Five-fold cross-validation with average precision as the optimization metric.
- Classification threshold selection using training data.
- Held-out test evaluation.
- Bootstrap confidence intervals using 2,000 resamples.
- Robustness and subgroup analyses.

### Classification Thresholds

| Model | Selected threshold |
|---|---:|
| Logistic Regression | 0.50 |
| Random Forest | 0.31 |
| XGBoost | 0.31 |

---

## 📈 Predictive Performance

### Held-Out Test Results

| Model | PR-AUC | ROC-AUC | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.754 | 0.756 | 0.627 | 0.638 |
| **Random Forest** | **0.816** | **0.838** | 0.831 | **0.715** |
| XGBoost | 0.800 | 0.812 | **0.847** | 0.704 |

Random Forest achieves the highest PR-AUC, ROC-AUC, and F1 score in this evaluation, while XGBoost achieves the highest recall.

PR-AUC is emphasized because identifying financially distressed municipalities is an important objective. ROC-AUC is also reported for comparability.

These metrics are evaluated on a case-control sample; precision and PR-AUC may differ under the actual population prevalence of fiscal distress.

### Bootstrap Confidence Intervals

The analysis uses 2,000 bootstrap resamples of the held-out test set to estimate uncertainty for:

- PR-AUC
- Recall
- Precision
- F1 score

Complete estimates and 95% percentile confidence intervals are provided in:

`results/bootstrap_confidence_intervals.csv`

---

## 🔍 Explainable Machine Learning

The XGBoost model is examined through multiple explainability methods.

### 1. Gain-Based Feature Importance

Provides a model-specific ranking of variables based on their contribution to tree splitting.

### 2. SHAP Feature Attribution

SHAP values quantify individual feature contributions to model predictions.

The analysis includes:

- Global SHAP summary plot.
- Mean absolute SHAP feature importance.
- Individual municipality waterfall explanations.

Numerical feature rankings are available in:

`results/shap_feature_importance.csv`

### 3. Pairwise Shapley Interactions

Pairwise feature interactions are estimated using the `shapiq` library and the k-SII interaction index with TreeSHAP-IQ.

The analysis evaluates interactions among financial, economic, and socio-demographic indicators, including interactions involving missingness indicators.

The complete ranked interaction results are available in:

`results/shap_interactions.csv`

These interaction values quantify joint predictive contributions, not causal relationships.

---

## 🧪 Robustness Analyses

Several supplementary analyses evaluate the sensitivity of model performance to alternative samples and feature configurations.

These include:

- Excluding heavily imputed financial features.
- Restricting the sample to municipalities reporting administrative deficit values.
- Comparing pre-COVID and COVID-era observations.
- Excluding 2016 observations.
- Excluding municipalities in the Islands macro-region.

Numerical results are provided in:

`results/robustness_checks.csv`

These analyses use alternative sample compositions and may not be directly comparable because the evaluation sets differ.

---

## 📉 Missing Data and Regional Analysis

The study investigates missing-data patterns across financial indicators, bankruptcy status, and Italian macro-regions.

A chi-square test is also conducted to examine the association between regional grouping and municipal distress status.

The repository includes:

- `results/missing_data_analysis.csv` — missingness percentages by region, feature, and distress status.
- `results/statistical_tests.csv` — chi-square test statistics.
- `results/regional_contingency_table.csv` — underlying regional contingency counts.

Missingness is considered an important methodological issue because financial reporting patterns may themselves carry predictive information.

---

## 📁 Repository Structure

```text
municipality-finance_health_Italy/
│
├── municipal_distress_pipeline.py
├── final_municipality_dataset_percap_with_indicators.csv
├── README.md
│
├── figures/
│   ├── class_balance.png
│   ├── decision_curve.png
│   ├── missing_data_analysis.png
│   ├── missing_data_by_region.png
│   ├── spending_comparison.png
│   ├── final_roc_curve.png
│   ├── final_pr_curve.png
│   ├── gain_importance.png
│   ├── shap_summary.png
│   ├── shap_interactions_top10.png
│   ├── waterfall_comparison_anonymous.png
│   └── waterfalls/
│
└── results/
    ├── model_performance.csv
    ├── bootstrap_confidence_intervals.csv
    ├── robustness_checks.csv
    ├── statistical_tests.csv
    ├── regional_contingency_table.csv
    ├── shap_feature_importance.csv
    ├── shap_interactions.csv
    ├── model_tuning.csv
    └── missing_data_analysis.csv
```

---

## 🔁 Reproducibility

The repository provides the Python analysis pipeline and saved numerical outputs to support independent inspection and reproduction.

### 1. Clone the Repository

```bash
git clone https://github.com/YasminMcodes/municipality-finance_health_Italy.git
cd municipality-finance_health_Italy
```

### 2. Set Up a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

The analysis uses Python libraries including:

- pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- shapiq
- Matplotlib and Seaborn
- SciPy

Exact compatible dependency versions should be documented in a `requirements.txt` file after environment verification.

### 4. Run the Analysis

The main analysis script is:

`municipal_distress_pipeline.py`

The script is intended to generate model evaluation outputs, robustness results, and explainability figures.

Before independent execution, verify that the dataset paths, package versions, and environment-specific settings are compatible with the repository structure.

### 5. Inspect the Results

Numerical outputs are available in the `results/` directory, while publication-oriented visualizations are available in `figures/`.

---

## ⚠️ Interpretation and Limitations

Several limitations should be considered when interpreting the results:

- The matched case-control design does not represent the natural prevalence of municipal fiscal distress in Italy.
- Missing financial indicators may reflect non-random reporting patterns.
- Predictive associations and SHAP attributions should not be interpreted as causal effects.
- Results from robustness analyses involving different subsamples are not necessarily directly comparable.
- Independent external validation would be necessary before using the models for operational early-warning decisions.

---

## 📚 Research Context

This repository accompanies academic research on the statistical analysis of social, economic, and financial indicators associated with municipal fiscal fragility in Italy.

The project emphasizes reproducible quantitative analysis, predictive evaluation, and interpretable machine learning.