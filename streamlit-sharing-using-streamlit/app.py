import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------
# Load all four persisted artifacts once. Streamlit re-runs this script on
# every widget interaction, so @st.cache_resource stops joblib.load() from
# re-reading the files on every click.
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    MODEL_DIR = "./model"

    regression_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib")
    )
    classification_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
    )
    clustering_artifact = joblib.load(
        os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib")
    )
    arm_artifact = joblib.load(
        os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
    )
    return regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact


regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact = load_artifacts()
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]


# -----------------------------------------------------------------------
# Read the categories each pipeline's encoder learned during training.
# The dropdowns below use these lists, so users can only pick values the
# model has seen. Any other value would make predict() fail.
# -----------------------------------------------------------------------
def get_categories(pipeline):
    categories = {}
    for _, step in pipeline.steps:
        if hasattr(step, "transformers_"):
            for _, transformer, columns in step.transformers_:
                encoder = transformer.steps[-1][1] if hasattr(transformer, "steps") else transformer
                if hasattr(encoder, "categories_"):
                    for column, values in zip(columns, encoder.categories_):
                        categories[column] = [str(v) for v in values]
    return categories


def pick(label, categories, column, default, key):
    """Dropdown built from the trained categories. Falls back to a text box
    only if the column was not found in the encoder."""
    options = categories.get(column)
    if options:
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=index, key=key)
    return st.text_input(label, default, key=key)


reg_cats = get_categories(regression_pipeline)
clf_cats = get_categories(classification_pipeline)

st.set_page_config(page_title="BBT 4206 Model Validation Demos", page_icon="\U0001F4CA", layout="centered")
st.title("BBT 4206 -- Model Validation Demos")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "SME Revenue Regressor", "SME Credit Risk Classifier",
    "Mall Customer Segmenter", "E-commerce Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression
# -----------------------------------------------------------------------
with tab_regression:
    with st.form("regression_form"):
        sector = pick("Sector", reg_cats, "sector", "Technology", "reg_sector")
        county = pick("County", reg_cats, "county", "Nairobi", "reg_county")
        quarter = pick("Quarter", reg_cats, "quarter", "Q3", "reg_quarter")
        business_age_years = st.number_input("Business Age (years)", value=4.5)
        num_employees = st.number_input("Number of Employees", value=12)
        owner_education_level = pick(
            "Owner Education Level", reg_cats, "owner_education_level", "Secondary", "reg_edu")
        monthly_marketing_spend_kes = st.number_input("Monthly Marketing Spend (KES)", value=45000)
        avg_daily_foot_traffic = st.number_input(
            "Avg Daily Foot Traffic (leave 0 if not applicable)", value=0)
        online_presence_score = st.number_input("Online Presence Score", value=78.0)
        loan_amount_kes = st.number_input("Loan Amount (KES)", value=250000)
        interest_rate_pct = st.number_input("Interest Rate (%)", value=14.5)
        inflation_rate_pct = st.number_input("Inflation Rate (%)", value=6.5)
        competitor_density = st.number_input("Competitor Density", value=5)
        customer_satisfaction_score = st.number_input("Customer Satisfaction Score", value=4.2)
        submitted = st.form_submit_button("Predict Revenue")

    if submitted:
        payload = {
            "sector": sector, "county": county, "quarter": quarter,
            "business_age_years": business_age_years, "num_employees": num_employees,
            "owner_education_level": owner_education_level,
            "monthly_marketing_spend_kes": monthly_marketing_spend_kes,
            "avg_daily_foot_traffic": avg_daily_foot_traffic or None,
            "online_presence_score": online_presence_score,
            "loan_amount_kes": loan_amount_kes, "interest_rate_pct": interest_rate_pct,
            "inflation_rate_pct": inflation_rate_pct,
            "competitor_density": competitor_density,
            "customer_satisfaction_score": customer_satisfaction_score,
        }
        input_row = pd.DataFrame([payload])
        predicted_log_revenue = regression_pipeline.predict(input_row)[0]
        predicted_revenue_kes = float(np.expm1(predicted_log_revenue))
        st.success(f"Predicted monthly revenue: KES {predicted_revenue_kes:,.2f}")

# -----------------------------------------------------------------------
# 2. Classification
# -----------------------------------------------------------------------
with tab_classification:
    with st.form("classification_form"):
        sector2 = pick("Sector", clf_cats, "sector", "Retail", "clf_sector")
        county2 = pick("County", clf_cats, "county", "Nairobi", "clf_county")
        quarter2 = pick("Quarter", clf_cats, "quarter", "Q2", "clf_quarter")
        loan_purpose = pick("Loan Purpose", clf_cats, "loan_purpose", "Working Capital", "clf_purpose")
        owner_education_level2 = pick(
            "Owner Education Level", clf_cats, "owner_education_level", "Secondary", "oel2")
        business_age_years2 = st.number_input("Business Age (years)", value=2.1, key="bay2")
        num_employees2 = st.number_input("Number of Employees", value=4, key="ne2")
        monthly_revenue_kes = st.number_input("Monthly Revenue (KES)", value=850000)
        loan_amount_kes2 = st.number_input("Loan Amount (KES)", value=150000, key="lak2")
        loan_term_months = st.number_input("Loan Term (months)", value=12)
        interest_rate_pct2 = st.number_input("Interest Rate (%)", value=16.5, key="irp2")
        collateral_value_kes = st.number_input("Collateral Value (leave 0 if unsecured)", value=0)
        debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", value=0.62)
        credit_score = st.number_input("Credit Score", value=590)
        num_previous_loans = st.number_input("Number of Previous Loans", value=1)
        missed_payments_count = st.number_input("Missed Payments Count", value=3)
        online_presence_score2 = st.number_input("Online Presence Score", value=40.0, key="ops2")
        customer_satisfaction_score2 = st.number_input(
            "Customer Satisfaction Score", value=3.4, key="css2")
        competitor_density2 = st.number_input("Competitor Density", value=6, key="cd2")
        submitted2 = st.form_submit_button("Predict Credit Risk")

    if submitted2:
        payload = {
            "sector": sector2, "county": county2, "quarter": quarter2,
            "loan_purpose": loan_purpose, "owner_education_level": owner_education_level2,
            "business_age_years": business_age_years2, "num_employees": num_employees2,
            "monthly_revenue_kes": monthly_revenue_kes, "loan_amount_kes": loan_amount_kes2,
            "loan_term_months": loan_term_months, "interest_rate_pct": interest_rate_pct2,
            "collateral_value_kes": collateral_value_kes or None,
            "debt_to_income_ratio": debt_to_income_ratio, "credit_score": credit_score,
            "num_previous_loans": num_previous_loans,
            "missed_payments_count": missed_payments_count,
            "online_presence_score": online_presence_score2,
            "customer_satisfaction_score": customer_satisfaction_score2,
            "competitor_density": competitor_density2,
        }
        input_row = pd.DataFrame([payload])
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_
        st.success(f"Predicted credit risk category: {predicted_class}")
        st.write({str(label): round(float(prob), 4)
                  for label, prob in zip(class_labels, predicted_probabilities)})

# -----------------------------------------------------------------------
# 3. Clustering
# -----------------------------------------------------------------------
with tab_clustering:
    with st.form("clustering_form"):
        age = st.number_input("Age", min_value=1, max_value=100, value=27)
        annual_income_k = st.number_input("Annual Income (k$)", min_value=0.0, value=45.0)
        spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=72)
        submitted3 = st.form_submit_button("Predict Segment")

    if submitted3:
        input_row = pd.DataFrame([{
            "Age": age,
            "Annual Income (k$)": annual_income_k,
            "Spending Score (1-100)": spending_score,
        }])
        scaled_input = clustering_scaler.transform(input_row)
        cluster = int(clustering_kmeans.predict(scaled_input)[0])
        st.success(f"Assigned segment: Cluster {cluster}")

# -----------------------------------------------------------------------
# 4. Association Rule Mining
# -----------------------------------------------------------------------
with tab_arm:
    with st.form("arm_form"):
        cart_text = st.text_input("Cart (comma-separated SKUs)", "H06, H07")
        top_n = st.number_input("Top N Recommendations", min_value=1, value=3)
        submitted4 = st.form_submit_button("Get Recommendations")

    if submitted4:
        cart = {sku.strip() for sku in cart_text.split(",") if sku.strip()}
        candidates = {}
        for antecedent, consequent_list in recommendation_lookup.items():
            if antecedent.issubset(cart):
                for sku, confidence, lift in consequent_list:
                    if sku not in cart:
                        if sku not in candidates or confidence > candidates[sku][0]:
                            candidates[sku] = (confidence, lift)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:int(top_n)]
        if not ranked:
            st.warning("No recommendations for this cart -- try a SKU combination seen in the training orders.")
        else:
            for sku, (confidence, lift) in ranked:
                st.write(f"**{sku_lookup.loc[sku, 'product_name']}** ({sku}) "
                         f"-- confidence {confidence:.1%}, lift {lift:.2f}")