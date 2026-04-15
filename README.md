# 🏛️ Municipal Fiscal Distress Prediction — Italy 2016–2023

A statistical analysis of social, economic, and financial indicators 
impacting municipal fiscal fragility in Italy, using an explainable 
machine learning framework.



---

## 🎯 Research Objective

This project identifies the **structural determinants of municipal fiscal 
fragility** in Italy over the 2016–2023 post-harmonization period. 
Beyond prediction, the framework delivers interpretable structural insight 
through SHAP-based decomposition, providing central authorities and 
policymakers with transparent, data-driven tools to identify at-risk 
municipalities **before** formal insolvency procedures are activated.

Three testable hypotheses are examined:

- **H1:** Administrative deficits and cashier advances are the dominant 
  drivers of fiscal distress, independently of debt levels
- **H2:** Education spending acts as a structural protective factor, 
  while welfare expenditure functions as a contextual risk segmenter
- **H3:** Socio-demographic fragility — proxied by low educational 
  attainment and population decline — amplifies distress risk 
  independently of financial indicators

---

## 📊 Data Sources

| Source | Content |
|--------|---------|
| [Open Bilanci](https://openbilanci.it/) | Municipal budget and financial data (2016–2022) |
| [Banca Dati Ca' Foscari](https://www.bancadaticomunicf.eu/public/elen_comu.php) | Official registry of municipal fiscal distress procedures |
| [ISTAT](https://www.istat.it/) | Socio-demographic indicators |

### Sample Design

The dataset follows a **matched case-control design**:
- All municipalities that formally entered fiscal distress 
  (*dissesto* or *riequilibrio*) between 2016–2023 were identified
- Each was matched to a non-distressed municipality on geographic 
  location, population size, and structural characteristics
- Final sample: **462 municipalities** (~42% distressed by design)

---

## 🧠 ML Pipeline

### Models
| Model | Strategy | Threshold |
|-------|----------|-----------|
| Logistic Regression | Linear Baseline | t = 0.50 |
| Random Forest | Safety Net (Max Recall) | t = 0.40 |
| XGBoost | Balanced Expert (Best F1) | t = 0.15 |

### Results (Held-Out Test Set)
| Model | PR-AUC | ROC-AUC | F1 | Recall |
|-------|--------|---------|-----|--------|
| Logistic Regression | 0.693 | 0.710 | 0.586 | 0.492 |
| Random Forest | 0.783 | 0.821 | 0.655 | 0.644 |
| **XGBoost** | **0.811** | **0.841** | **0.737** | **0.831** |

> PR-AUC is the primary metric given the case-control sample design.
> ROC-AUC is reported for comparability with the broader literature.

### Explainability Framework
Three-tiered explainability applied to XGBoost:
1. **Gain-based Feature Importance** — global structural ranking
2. **SHAP Summary Plot** — direction and magnitude of each driver
3. **SHAP Waterfall Plots** — individual municipality diagnostics

---

## 📁 Repository Structure
