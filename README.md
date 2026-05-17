# 🤱 KAP-BPCR — Predicting Birth Preparedness and Complication Readiness

> **Binary classification of maternal health readiness among pregnant women in Ethiopia**  
> Course: Design and Analysis of Algorithms · Instructor: Dr. Mekia Shigute Gaso

---

## 📋 Overview

Ethiopia has one of the highest maternal mortality rates in the world — approximately **401 deaths per 100,000 live births** (EDHS 2019). Birth Preparedness and Complication Readiness (BPCR) is a WHO-recommended strategy that directly reduces delays in reaching emergency obstetric care.

This project builds a machine learning pipeline to predict whether a pregnant woman has **good BPCR practice** based on survey data — so healthcare workers can target counseling early.

Women who complete all 4 BPCR steps are **3× more likely** to receive timely obstetric care:
1. Choose a birth location in advance
2. Save money for delivery costs
3. Arrange reliable transport
4. Identify a potential blood donor

> ⚠️ The model is a decision-support tool. It does **not** replace clinical judgment.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | KAP-BPCR Survey, Ethiopia |
| Respondents | 411 pregnant women |
| Raw features | 73 columns |
| Target | `good_bpcr_practice` (BPCR score ≥ 5 out of 8) |
| Class distribution | 80.8% good · 19.2% poor |

**Feature groups:**
- **Demographics** — age, education, income, occupation, family size
- **Knowledge** — danger signs in pregnancy / delivery / postpartum
- **Attitudes** — 5 Likert-scale items on BPCR components
- **Practices** — 8 binary items (TARGET, removed from model inputs to prevent leakage)

**Missingness types (Rubin's taxonomy):**
- `src_*` (~59%) — **MNAR**: missing when `heard_bpcr = no`
- `kbc_*` (~45%) — **MAR**: conditional on `know_bpcr_components`
- `income` (~2%) — **MCAR**: random non-response

---

## 🔬 Pipeline

```
Raw data (411 × 73)
    │
    ├── Stage 1: Data Cleaning
    │     ├── Missingness classification (MCAR / MAR / MNAR)
    │     ├── Imputation comparison: Mean · Median · KNN · Iterative MICE ✓
    │     └── Outlier detection: Isolation Forest + Robust Z-score (MAD)
    │
    ├── Stage 2: Feature Engineering  [44 → 83 features]
    │     ├── Domain-inspired indices (health_literacy_idx, vulnerability_index)
    │     ├── Interaction features (age × education, income / family_size)
    │     ├── Statistical transforms (Box-Cox, Quantile-Normal, log1p)
    │     ├── Autoencoder latent features (43 → 8 bottleneck, MLP)
    │     └── Stability analysis (top-K across 10 random seeds)
    │
    ├── Stage 3: Feature Selection  [83 → 28 → 14 features]
    │     ├── Mutual Information (averaged over 10 seeds)
    │     ├── SHAP (TreeExplainer, mean |SHAP value|)
    │     ├── RFE (Recursive Feature Elimination, Logistic Regression)
    │     ├── Stability Selection (multi-seed top-K voting)
    │     └── Consensus: feature included if selected by ≥ 3 / 4 methods
    │
    ├── Stage 4: Class Imbalance
    │     ├── No resampling (baseline)
    │     ├── Class weight = balanced
    │     ├── SMOTE (standard) ✓
    │     ├── SMOTE-ENN
    │     └── Adaptive Noise SMOTE (novel contribution)
    │
    ├── Stage 5: Modern ML Models
    │     ├── HGBC · XGBoost · LightGBM · CatBoost · MLPClassifier
    │     └── Novel: Multi-Seed Weighted Ensemble (seeds 11, 22, 33)
    │
    ├── Stage 6: Robust Evaluation
    │     └── Stratified 5-fold CV · mean ± std · fold variance analysis
    │
    ├── Stage 7: Explainability
    │     └── SHAP global importance + local patient-level explanations
    │
    ├── Stage 8: Error Analysis
    │     └── False Positive / False Negative patient profiles
    │
    ├── Stage 9: Probability Calibration
    │     └── Platt Sigmoid · Isotonic · Brier score comparison
    │
    ├── Stage 10: Learning Curves
    │     └── Overfitting / underfitting / saturation diagnosis
    │
    └── Stage 11: Threshold Optimization
          └── Sweep 0.05–0.95 · optimize for Recall ≥ 0.85
```

---

## 🤖 Models

| Model | F1 | Recall | AUC | Train (s) | Type |
|---|---|---|---|---|---|
| **CatBoost ★** | **0.8936** | 0.9126 | **0.7738** | 0.471 | Ordered boosting |
| XGBoost | 0.8882 | 0.9338 | 0.7444 | 0.124 | Level-wise boosting |
| NeuralNetwork | 0.8901 | **0.9758** | 0.7144 | 0.013 | MLP (64→32) |
| LightGBM | 0.8784 | 0.9038 | 0.7400 | **0.062** | Leaf-wise boosting |
| HGBC | 0.8578 | 0.8645 | 0.7429 | 0.126 | Histogram boosting |

> ★ CatBoost = best F1 and AUC · LightGBM = Pareto-optimal for deployment (speed vs accuracy)

---

## 💡 Novel Contributions

### 1. Multi-Seed Weighted Ensemble
Three models (HGBC, XGBoost, LightGBM) trained on seeds 11, 22, 33. Predicted probabilities averaged with weights:

```python
weights = {'HGBC': 1.0, 'XGBoost': 1.25, 'LightGBM': 1.25}
```

This explicitly averages over initialization instability — not just bootstrap variance like standard bagging. In medical datasets, a single bad split can produce systematically biased conclusions.

### 2. Adaptive Noise SMOTE
Standard SMOTE creates synthetic points exactly on line segments between minority class pairs, producing artificial patterns the model memorizes. Our method adds Gaussian noise proportional to local density:

```python
synthetic = α × x_i + (1-α) × x_j + N(0, noise_scale × feature_std)
```

This breaks synthetic artifacts and improves generalization on real patient profiles.

---

## 📈 Key Results

```
Feature trajectory:   44 raw → 83 after engineering → 28 consensus → 14 strict
Best imputation:      Iterative MICE (Bayesian Ridge)
Best imbalance:       Standard SMOTE  (F1 ≈ 0.96 ± 0.009, Recall ≈ 0.97)
Best model (F1/AUC):  CatBoost + Sigmoid calibration
Best for deployment:  LightGBM (same F1, 7.5× faster than CatBoost)
Optimal threshold:    Tuned for Recall ≥ 0.85 (reduces FN by 20–30%)
```

**Algorithm complexity:**

| Model | Training | Prediction |
|---|---|---|
| HGBC / XGBoost | O(T × n × d × b) | O(T × d) |
| LightGBM | O(T × n × k) | O(T × k) — faster |
| CatBoost | O(T × n × d) | O(T × d) |
| MLP | O(E × n × H²) | O(H²) — fastest inference |

`T` = trees, `n` = samples (411), `d` = features (14–28), `b` = bins, `k` = leaves, `E` = epochs, `H` = hidden size

---

## 🏥 Clinical Findings

**Top predictive features (SHAP consensus):**
`knowledge_score` · `education` · `info_diversity` · `kbc_count` · `att_total` · `health_literacy_idx`

**False Negative profile** (missed poor readiness — critical):
- High education but very low income
- Good knowledge score but very young age
- Contradictory signals → require manual clinical review

**False Positive profile** (predicted good, actually poor):
- High `knowledge_score` (knows BPCR theory)
- Low income: cannot afford transport or save money

> **Key insight: Knowledge ≠ Practice.** Telling a woman about BPCR is not enough. Healthcare systems must also address resource barriers — transport, savings, blood donor access.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/kap-bpcr.git
cd kap-bpcr
pip install -r requirements.txt
```

**Requirements:**
```
numpy pandas matplotlib seaborn scipy scikit-learn
xgboost lightgbm catboost shap joblib openpyxl
```

---

## 🚀 Usage

```bash
# Run full pipeline
jupyter notebook KAP_BPCR_FINAL_clean_structured.ipynb

# Or run stages individually
python src/stage1_cleaning.py
python src/stage2_engineering.py
python src/stage3_selection.py
python src/stage5_models.py
```

**Data setup:**
Place `KAP_BPCR_Data.xlsx` in the project root or update `DATA_CANDIDATES` in the notebook.

---

## 📁 Project Structure

```
kap-bpcr/
├── KAP_BPCR_FINAL_clean_structured.ipynb  # Main pipeline notebook
├── KAP_BPCR_Data.xlsx                      # Dataset (not included)
├── pipeline_state.pkl                      # Saved pipeline state
├── report/
│   └── BPCR_Report_Redesigned.docx         # Full project report
├── presentation/
│   └── KAP_BPCR_Final_Presentation.pptx   # Defense slides (19 slides)
├── figures/
│   ├── fig_demographics.png
│   ├── fig_kap_clinical_subgroup_bias.png
│   ├── fig_kap_roc_pr_confusion.png
│   └── fig_kap_error_profile_deep.png
└── README.md
```

---

## 📐 Evaluation Protocol

- **Cross-validation:** Stratified 5-fold, fixed `random_state=42`
- **Metrics:** Accuracy · Precision · Recall · F1 · AUC-ROC
- **Reporting:** mean ± std across folds (fold variance analyzed separately)
- **Calibration:** Brier score with Sigmoid and Isotonic methods
- **Threshold:** Swept 0.05–0.95, optimized for clinical Recall ≥ 0.85

> A model with high mean F1 but high fold variance is **not clinically reliable** even if the average looks good.

---

## 🎯 Final Recommendation

```
Model:       CatBoost
Calibration: Sigmoid (Platt scaling)  →  best Brier score = 0.142
Threshold:   Optimized for Recall ≥ 0.85

For frequent retraining:  LightGBM + Sigmoid
                          (near-identical F1, 7.5× faster)
```

---

## 📚 References

- Ethiopian Demographic and Health Survey (EDHS) 2019
- WHO guidelines on Birth Preparedness and Complication Readiness
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)
- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
- Ke et al. (2017) — LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- Prokhorenkova et al. (2018) — CatBoost: unbiased boosting with categorical features
