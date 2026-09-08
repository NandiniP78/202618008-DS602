import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, chi2_contingency, f_oneway
from statsmodels.graphics.gofplots import qqplot

st.set_page_config(page_title="Medical Insurance Cost Analysis", layout="wide")

# ---------- Data Loading ----------
@st.cache_data
def load_data():
    df = pd.read_csv('data/insurance.csv')
    return df

@st.cache_data
def preprocess(df):
    df_model = df.copy()
    df_model['sex'] = df_model['sex'].map({'male': 1, 'female': 0})
    df_model['smoker'] = df_model['smoker'].map({'yes': 1, 'no': 0})
    df_model = pd.get_dummies(df_model, columns=['region'], drop_first=True)
    bool_cols = df_model.select_dtypes(include='bool').columns
    df_model[bool_cols] = df_model[bool_cols].astype(int)
    return df_model

# ---------- Model Training (cached so it only trains once) ----------
@st.cache_resource
def train_model(df_model):
    X = df_model.drop('charges', axis=1)
    y = df_model['charges']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X, y

df = load_data()
df_model = preprocess(df)
model, X_full, y_full = train_model(df_model)

st.title("🏥 Medical Insurance Cost — Analysis Dashboard")

tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "🔮 Live Prediction & Diagnostics"
])

with tab1:
    st.header("Data Exploration")

    # ---------- Sidebar Filters ----------
    st.sidebar.header("Filters (Tab 1)")

    age_range = st.sidebar.slider(
        "Age Range", int(df['age'].min()), int(df['age'].max()),
        (int(df['age'].min()), int(df['age'].max()))
    )
    bmi_range = st.sidebar.slider(
        "BMI Range", float(df['bmi'].min()), float(df['bmi'].max()),
        (float(df['bmi'].min()), float(df['bmi'].max()))
    )
    charges_range = st.sidebar.slider(
        "Charges Range", float(df['charges'].min()), float(df['charges'].max()),
        (float(df['charges'].min()), float(df['charges'].max()))
    )
    region_filter = st.sidebar.multiselect(
        "Region", options=df['region'].unique(), default=list(df['region'].unique())
    )
    smoker_filter = st.sidebar.multiselect(
        "Smoker", options=df['smoker'].unique(), default=list(df['smoker'].unique())
    )
    sex_filter = st.sidebar.multiselect(
        "Sex", options=df['sex'].unique(), default=list(df['sex'].unique())
    )

    # ---------- Apply Filters ----------
    filtered_df = df[
        (df['age'].between(*age_range)) &
        (df['bmi'].between(*bmi_range)) &
        (df['charges'].between(*charges_range)) &
        (df['region'].isin(region_filter)) &
        (df['smoker'].isin(smoker_filter)) &
        (df['sex'].isin(sex_filter))
    ]

    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records** based on current filters.")

    # ---------- Summary Statistics ----------
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)

    # ---------- Reactive Plots ----------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Charges")
        fig = px.histogram(filtered_df, x='charges', nbins=40, color='smoker',
                            marginal='box', title="Charges Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Age vs. Charges")
        fig2 = px.scatter(filtered_df, x='age', y='charges', color='smoker',
                           size='bmi', hover_data=['region', 'children'],
                           title="Age vs Charges (bubble size = BMI)")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("BMI vs. Charges")
    fig3 = px.scatter(filtered_df, x='bmi', y='charges', color='smoker',
                       trendline="ols", title="BMI vs Charges with Trendline")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation Matrix (Numerical Features)")
    numeric_cols = ['age', 'bmi', 'children', 'charges']
    corr = filtered_df[numeric_cols].corr()
    fig4, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig4)

    st.subheader("Charges by Region")
    fig5 = px.box(filtered_df, x='region', y='charges', color='region',
                   title="Charges Distribution by Region")
    st.plotly_chart(fig5, use_container_width=True)

with tab2:
    st.header("Hypothesis Testing Lab")
    alpha = 0.05

    st.markdown("### Test 1: Compare a Numerical Metric Across a Categorical Factor")

    categorical_cols = ['sex', 'smoker', 'region']
    numerical_cols = ['age', 'bmi', 'children', 'charges']

    col1, col2 = st.columns(2)
    with col1:
        cat_var = st.selectbox("Select Categorical Factor", categorical_cols)
    with col2:
        num_var = st.selectbox("Select Numerical Metric", numerical_cols, index=3)

    groups = df[cat_var].unique()
    group_data = [df[df[cat_var] == g][num_var] for g in groups]

    st.write(f"**H0:** No significant difference in `{num_var}` across `{cat_var}` groups.")
    st.write(f"**H1:** There IS a significant difference in `{num_var}` across `{cat_var}` groups.")

    if len(groups) == 2:
        # Normality check
        p_norm = [shapiro(g)[1] for g in group_data]
        p_levene = levene(*group_data)[1]

        st.write(f"Shapiro-Wilk p-values: {dict(zip(groups, [round(p,4) for p in p_norm]))}")
        st.write(f"Levene's Test p-value: {p_levene:.4f}")

        if all(p > alpha for p in p_norm):
            stat, p_value = ttest_ind(*group_data, equal_var=(p_levene > alpha))
            test_name = "Two-Sample t-test"
        else:
            stat, p_value = mannwhitneyu(*group_data, alternative='two-sided')
            test_name = "Mann-Whitney U test"

        st.write(f"**Test used:** {test_name}")
        st.write(f"**Statistic:** {stat:.4f} | **p-value:** {p_value:.4g}")

    else:
        stat, p_value = f_oneway(*group_data)
        test_name = "One-Way ANOVA"
        st.write(f"**Test used:** {test_name}")
        st.write(f"**F-statistic:** {stat:.4f} | **p-value:** {p_value:.4g}")

    if p_value < alpha:
        st.success(f"✅ Reject H0 at α = 0.05 — there IS a significant difference in `{num_var}` across `{cat_var}` groups.")
    else:
        st.warning(f"❌ Fail to Reject H0 at α = 0.05 — no significant difference detected.")

    # Visual support
    fig = px.box(df, x=cat_var, y=num_var, color=cat_var, title=f"{num_var} by {cat_var}")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------- Chi-Square Test Section ----------
    st.markdown("### Test 2: Chi-Square Test (Association Between Two Categorical Variables)")

    col3, col4 = st.columns(2)
    with col3:
        cat_var1 = st.selectbox("Categorical Variable 1", categorical_cols, index=1, key='cat1')
    with col4:
        cat_var2 = st.selectbox("Categorical Variable 2", categorical_cols, index=2, key='cat2')

    if cat_var1 == cat_var2:
        st.info("Please select two different categorical variables.")
    else:
        contingency = pd.crosstab(df[cat_var1], df[cat_var2])
        chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency)

        st.write(f"**H0:** `{cat_var1}` and `{cat_var2}` are independent (not linked).")
        st.write(f"**H1:** `{cat_var1}` and `{cat_var2}` are associated (linked).")
        st.write(f"**Chi-Square Statistic:** {chi2_stat:.4f} | **p-value:** {p_chi2:.4g} | **dof:** {dof}")

        if p_chi2 < alpha:
            st.success(f"✅ Reject H0 at α = 0.05 — `{cat_var1}` and `{cat_var2}` ARE significantly associated.")
        else:
            st.warning(f"❌ Fail to Reject H0 at α = 0.05 — no significant association detected.")

        st.dataframe(contingency, use_container_width=True)

with tab3:
    st.header("Live Prediction & Diagnostics")

    st.subheader("Enter Applicant Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_input = st.slider("Age", 18, 100, 30)
        bmi_input = st.number_input("BMI", 10.0, 60.0, 25.0)

    with col2:
        children_input = st.slider("Children", 0, 5, 0)
        sex_input = st.selectbox("Sex", ["male", "female"])

    with col3:
        smoker_input = st.selectbox("Smoker", ["yes", "no"])
        region_input = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("Predict Charges"):
        # Build input row matching training columns exactly
        input_dict = {
            'const': 1.0,
            'age': age_input,
            'sex': 1 if sex_input == 'male' else 0,
            'bmi': bmi_input,
            'children': children_input,
            'smoker': 1 if smoker_input == 'yes' else 0,
            'region_northwest': 1 if region_input == 'northwest' else 0,
            'region_southeast': 1 if region_input == 'southeast' else 0,
            'region_southwest': 1 if region_input == 'southwest' else 0,
        }
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X_full.columns, fill_value=0)

        prediction = model.get_prediction(input_df)
        summary_frame = prediction.summary_frame(alpha=0.05)

        pred_value = summary_frame['mean'].values[0]
        ci_low, ci_high = summary_frame['mean_ci_lower'].values[0], summary_frame['mean_ci_upper'].values[0]
        pi_low, pi_high = summary_frame['obs_ci_lower'].values[0], summary_frame['obs_ci_upper'].values[0]

        st.success(f"### Predicted Charge: ${pred_value:,.2f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("95% Confidence Interval (mean estimate)",
                      f"${ci_low:,.0f} — ${ci_high:,.0f}")
        with col_b:
            st.metric("95% Prediction Interval (individual estimate)",
                      f"${pi_low:,.0f} — ${pi_high:,.0f}")

        st.caption(
            "The Confidence Interval reflects uncertainty in the *average* charge for someone "
            "with these characteristics. The Prediction Interval is wider because it reflects "
            "uncertainty for *this specific individual's* actual charge."
        )

    st.divider()

    # ---------- Residual Diagnostics (based on full trained model) ----------
    st.subheader("Model Residual Diagnostics")

    fitted_values = model.fittedvalues
    residuals = model.resid

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        st.markdown("**Residuals vs. Fitted Values**")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(fitted_values, residuals, alpha=0.4, color='steelblue')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        st.pyplot(fig1)

    with diag_col2:
        st.markdown("**Q-Q Plot of Residuals**")
        fig2 = qqplot(residuals, line='45', fit=True)
        st.pyplot(fig2)

    st.caption(
        "These diagnostic plots reflect the overall trained model's residual behavior "
        "(not specific to the single prediction above)."
    )