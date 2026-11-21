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
        os.path.join(MODEL_DIR, "20261121_170647_ridge_regressor_for_farm_income.joblib")
    )
    classification_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "20261121_171040_svc_classifier_for_invoice_risk.joblib")
    )
    clustering_artifact = joblib.load(
        os.path.join(MODEL_DIR, "20261121_170750_kmeans_farmer_segmentation.joblib")
    )
    arm_artifact = joblib.load(
        os.path.join(MODEL_DIR, "20261121_170352_delivery_recommendation_rules.joblib")
    )
    return regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact


regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact = load_artifacts()
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
clustering_features = clustering_artifact["features"]
recommendation_lookup = arm_artifact["recommendation_lookup"]
item_lookup = arm_artifact["item_lookup"]

CLUSTER_NAMES = {
    0: "Low-access smallholders",
    1: "Established smallholders with good market access",
    2: "Commercial farmers",
    3: "High-value niche farmers",
}


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


def or_missing(value):
    """A 0 in a 'leave 0 if not applicable' box means the value is missing.
    The pipeline's own imputer handles missing values, as it did in training."""
    return np.nan if value == 0 else value


reg_cats = get_categories(regression_pipeline)
clf_cats = get_categories(classification_pipeline)

st.set_page_config(page_title="Group D2 Model Validation Demos", page_icon="\U0001F4CA", layout="centered")
st.title("Group D2 -- Model Validation Demos")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "Farm Income Regressor", "Invoice Risk Classifier",
    "Farmer Segmenter", "Delivery Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression: predict annual farm income (KES)
# -----------------------------------------------------------------------
with tab_regression:
    with st.form("regression_form"):
        crop_type = pick("Crop Type", reg_cats, "crop_type", "Tea", "reg_crop")
        county = pick("County", reg_cats, "county", "Uasin Gishu", "reg_county")
        irrigation_type = pick("Irrigation Type", reg_cats, "irrigation_type", "Drip", "reg_irrigation")
        cooperative_member = pick("Cooperative Member", reg_cats, "cooperative_member", "No", "reg_coop")
        farm_size_acres = st.number_input("Farm Size (acres)", value=7.26)
        years_farming_experience = st.number_input("Years of Farming Experience", value=10.0)
        num_workers_employed = st.number_input("Number of Workers Employed", value=1)
        rainfall_mm = st.number_input("Rainfall (mm)", value=1345.0)
        soil_quality_score = st.number_input("Soil Quality Score (leave 0 if unknown)", value=0.0)
        market_distance_km = st.number_input("Market Distance (km)", value=5.7)
        irrigation_maintenance_cost_kes = st.number_input(
            "Irrigation Maintenance Cost (KES, leave 0 if no irrigation)", value=20609.32)
        submitted = st.form_submit_button("Predict Annual Farm Income")

    if submitted:
        input_row = pd.DataFrame([{
            "years_farming_experience": years_farming_experience,
            "soil_quality_score": or_missing(soil_quality_score),
            "num_workers_employed": num_workers_employed,
            "rainfall_mm": rainfall_mm,
            "market_distance_km": market_distance_km,
            "farm_size_acres": farm_size_acres,
            "irrigation_maintenance_cost_kes": or_missing(irrigation_maintenance_cost_kes),
            "crop_type": crop_type,
            "county": county,
            "irrigation_type": irrigation_type,
            "cooperative_member": cooperative_member,
        }])
        # The model was trained on log1p(income), so expm1 turns it back into KES.
        predicted_log_income = regression_pipeline.predict(input_row)[0]
        predicted_income_kes = float(np.expm1(predicted_log_income))
        st.success(f"Predicted annual farm income: KES {predicted_income_kes:,.2f}")

# -----------------------------------------------------------------------
# 2. Classification: predict invoice payment risk
# -----------------------------------------------------------------------
with tab_classification:
    with st.form("classification_form"):
        client_industry = pick("Client Industry", clf_cats, "client_industry", "Technology", "clf_industry")
        client_credit_rating = pick("Client Credit Rating", clf_cats, "client_credit_rating", "Good", "clf_rating")
        client_company_size = pick("Client Company Size", clf_cats, "client_company_size", "Large", "clf_size")
        payment_method = pick("Payment Method", clf_cats, "payment_method", "Bank Transfer", "clf_method")
        has_dispute_history = pick("Has Dispute History", clf_cats, "has_dispute_history", "Yes", "clf_dispute")
        collateral_or_guarantee = pick(
            "Collateral or Guarantee", clf_cats, "collateral_or_guarantee", "No", "clf_collateral")
        payment_terms_days = st.number_input("Payment Terms (days)", value=45)
        invoice_amount_kes = st.number_input("Invoice Amount (KES)", value=256337.0)
        client_relationship_years = st.number_input("Client Relationship (years)", value=1.9)
        prior_late_payments_count = st.number_input("Prior Late Payments", value=1)
        discount_offered_pct = st.number_input("Discount Offered (%)", value=2.73)
        days_outstanding_avg_historical = st.number_input("Average Days Outstanding (historical)", value=22.2)
        guarantee_value_kes = st.number_input("Guarantee Value (KES, leave 0 if no guarantee)", value=0.0)
        economic_sector_growth_rate_pct = st.number_input("Economic Sector Growth Rate (%)", value=4.87)
        submitted2 = st.form_submit_button("Predict Payment Risk")

    if submitted2:
        input_row = pd.DataFrame([{
            "client_industry": client_industry,
            "client_credit_rating": client_credit_rating,
            "client_company_size": client_company_size,
            "payment_method": payment_method,
            "has_dispute_history": has_dispute_history,
            "collateral_or_guarantee": collateral_or_guarantee,
            "payment_terms_days": payment_terms_days,
            "invoice_amount_kes": invoice_amount_kes,
            "client_relationship_years": client_relationship_years,
            "prior_late_payments_count": prior_late_payments_count,
            "discount_offered_pct": discount_offered_pct,
            "days_outstanding_avg_historical": days_outstanding_avg_historical,
            "guarantee_value_kes": or_missing(guarantee_value_kes),
            "economic_sector_growth_rate_pct": economic_sector_growth_rate_pct,
        }])
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_
        st.success(f"Predicted payment risk category: {predicted_class}")
        st.write({str(label): round(float(prob), 4)
                  for label, prob in zip(class_labels, predicted_probabilities)})

# -----------------------------------------------------------------------
# 3. Clustering: assign a cooperative farmer to a segment
# -----------------------------------------------------------------------
with tab_clustering:
    with st.form("clustering_form"):
        c_farm_size = st.number_input("Farm Size (acres)", min_value=0.0, value=16.29, key="c_farm")
        c_yield = st.number_input("Annual Yield (tonnes)", min_value=0.0, value=35.9)
        c_income = st.number_input("Income per Acre (KES)", min_value=0.0, value=2000.0)
        c_experience = st.number_input("Years of Farming Experience", min_value=0.0, value=7.0, key="c_exp")
        c_access = st.number_input("Market Access Score", min_value=0.0, value=51.7)
        submitted3 = st.form_submit_button("Predict Segment")

    if submitted3:
        values = {
            "farm_size_acres": c_farm_size,
            "annual_yield_tonnes": c_yield,
            "income_per_acre_kes": c_income,
            "years_farming_experience": c_experience,
            "market_access_score": c_access,
        }
        input_row = pd.DataFrame([values])[clustering_features]
        scaled_input = clustering_scaler.transform(input_row)
        cluster = int(clustering_kmeans.predict(scaled_input)[0])
        st.success(f"Assigned segment: Cluster {cluster} -- {CLUSTER_NAMES.get(cluster, 'Unknown')}")

# -----------------------------------------------------------------------
# 4. Association Rule Mining: recommend food items for a cart
# -----------------------------------------------------------------------
with tab_arm:
    item_names = {item_id: str(item_lookup.loc[item_id, "item_name"]) for item_id in item_lookup.index}
    all_ids = sorted(item_names)

    with st.form("arm_form"):
        chosen_ids = st.multiselect(
            "Items in the cart",
            options=all_ids,
            default=["F01", "F03"],
            format_func=lambda item_id: f"{item_names[item_id]} ({item_id})",
        )
        top_n = st.number_input("Top N Recommendations", min_value=1, value=3)
        submitted4 = st.form_submit_button("Get Recommendations")

    if submitted4:
        cart = set(chosen_ids)
        if not cart:
            st.warning("Please pick at least one item for the cart.")
        else:
            candidates = {}
            for antecedent, consequent_list in recommendation_lookup.items():
                if antecedent.issubset(cart):
                    for item, confidence, lift in consequent_list:
                        if item not in cart:
                            if item not in candidates or confidence > candidates[item][0]:
                                candidates[item] = (confidence, lift)

            ranked = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)[:int(top_n)]
            if not ranked:
                st.warning("No recommendations for this cart. Try a different combination of items.")
            else:
                for item, (confidence, lift) in ranked:
                    st.write(f"**{item_names[item]}** ({item}) "
                             f"-- confidence {confidence:.1%}, lift {lift:.2f}")