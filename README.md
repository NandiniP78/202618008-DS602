# Medical Insurance Cost Prediction

An end-to-end data science project analyzing medical insurance charges through exploratory
data analysis, statistical hypothesis testing, and multiple linear regression — deployed as
an interactive Streamlit dashboard.

## Table of Contents
- [Dataset Summary](#dataset-summary)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Statistical Findings Summary](#statistical-findings-summary)
- [Dashboard Overview](#dashboard-overview)

---

## Dataset Summary

**Source:** [Medical Insurance Cost Dataset — Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset/data)

The dataset contains **1,338 records** of individual medical insurance beneficiaries in the US,
with the following features:

| Feature    | Type        | Description                                           |
|------------|-------------|--------------------------------------------------------|
| `age`      | Numerical   | Age of the primary beneficiary                        |
| `sex`      | Categorical | Gender (male / female)                                 |
| `bmi`      | Numerical   | Body Mass Index                                        |
| `children` | Numerical   | Number of dependents covered by insurance               |
| `smoker`   | Categorical | Smoking status (yes / no)                               |
| `region`   | Categorical | Residential region (northeast, northwest, southeast, southwest) |
| `charges`  | Numerical   | Individual medical costs billed by insurance (target variable) |

No missing values were found in the dataset. `charges` is right-skewed, with a small
proportion of high-cost outliers largely driven by smoking status.

---

## Project Structure
medical-insurance-cost-prediction/
│
├── data/
│ └── raw/
│ └── insurance.csv
│
├── notebooks/
│ └── insurance_analysis.ipynb # EDA, hypothesis testing, OLS model & diagnostics
│
├── app.py # Streamlit dashboard (3 tabs)
├── requirements.txt
├── README.md
└── .gitignore


---

## How to Run

This project runs locally via VS Code.

**1. Set up virtual environment**
```bash
python -m venv venv
```
Activate it:
```bash
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the dashboard**
```bash
streamlit run app.py
```
The app opens automatically at [http://localhost:8501](http://localhost:8501).

To stop the app, go to the terminal running Streamlit and press `Ctrl + C`.

---

## Statistical Findings Summary

### Exploratory Data Analysis
- `charges` is heavily right-skewed with high kurtosis, reflecting a small subset of very
  high-cost individuals — overwhelmingly smokers.
- `age` and `bmi` are roughly symmetric; `children` is discrete and right-skewed.
- Correlation among numerical features is generally weak (`age`, `bmi` show weak-to-moderate
  positive correlation with `charges`), but `smoker` status shows the strongest visual
  separation in scatter plots despite not appearing in the numeric correlation matrix.

### Hypothesis Test 1 — Smokers vs. Non-Smokers (Charges)
- **H0:** No difference in mean charges between smokers and non-smokers.
- Normality was violated (Shapiro-Wilk, p < 0.05) and/or variances were unequal (Levene's
  test), so a **Mann-Whitney U test** was used.
- **Result:** p < 0.05 → **Reject H0**. Smokers incur significantly higher medical charges
  than non-smokers.

### Hypothesis Test 2 — One-Way ANOVA (Charges Across Regions)
- **H0:** Mean charges are equal across all four regions.
- **Result:** F-statistic = 2.97, p = 0.031 → **Reject H0** at α = 0.05. Average charges differ
  significantly across regions, though the effect is modest (southeast tends to show the
  largest deviation). Note: ANOVA only indicates *some* group differs — a post-hoc test (e.g.,
  Tukey HSD) would be needed to identify which specific regions differ.

### Multiple Linear Regression (OLS)
- **Model fit:** R² = 0.751, Adjusted R² = 0.749 — the model explains ~75% of variance in charges.
- **Significant predictors (p < 0.05):** `age`, `bmi`, `children`, `smoker`,
  `region_southeast`, `region_southwest`.
- **Not significant:** `sex`, `region_northwest`.
- **Strongest effect by far:** `smoker` (β ≈ +23,850) — being a smoker is associated with
  roughly $23,850 higher charges, holding other variables constant.
- Other notable effects: each additional year of `age` (+$257), each unit of `bmi` (+$339),
  and each additional `child` (+$476) are all associated with higher charges.

### Diagnostic Checks (Gauss-Markov Assumptions)
- **Linearity / Homoscedasticity:** Violated — Residuals vs. Fitted plot shows distinct
  clustering and non-constant variance, primarily driven by the smoker/non-smoker split and
  likely nonlinear interactions.
- **Normality of Residuals:** Violated — Q-Q plot and Jarque-Bera test show right-skewed,
  non-normal residuals, consistent with the skewed `charges` distribution.
- **Multicollinearity:** Not a concern — all VIF values are well below 5 (highest is ~1.65 for
  `region_southeast`), confirming predictors are not redundant with each other.

**Overall takeaway:** Smoking status is the dominant driver of medical insurance charges, with
age, BMI, and number of children as secondary but statistically significant contributors. While
the linear model achieves a strong R² (~75%), violations of linearity, homoscedasticity, and
normality suggest a log-transformed target or non-linear model could improve both prediction
accuracy and the validity of statistical inference.

---

## Dashboard Overview

The Streamlit app (`app.py`) is organized into three tabs:

1. **Data Exploration** — Sidebar filters (age, BMI, charges range, region, smoker, sex) with
   reactive Plotly/Seaborn visualizations and live summary statistics.
2. **Hypothesis Testing Lab** — Select any categorical factor and numerical metric to
   automatically run the appropriate test (t-test, Mann-Whitney U, or ANOVA) with live
   conclusions, plus a Chi-Square test for association between categorical variables.
3. **Live Prediction & Diagnostics** — Enter applicant details to get a real-time charge
   prediction with 95% confidence and prediction intervals, alongside the model's residual
   diagnostic plots.