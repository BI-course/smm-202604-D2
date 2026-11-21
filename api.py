"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- Farm annual income prediction (Ridge)
    2. Classification   -- Invoice payment risk category (SVC)
    3. Clustering       -- Cooperative farmer segment assignment (K-Means, k=4)
    4. Association Rule -- Food delivery item recommendation from a cart
       Mining (ARM)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------

MODEL_DIR = "./model"

# 1. Regression: Ridge pipeline trained on log1p(annual farm income)
regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "20261121_170647_ridge_regressor_for_farm_income.joblib")
)

# 2. Classification: SVC pipeline (preprocessing + SMOTE + model)
classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "20261121_171040_svc_classifier_for_invoice_risk.joblib")
)

# 3. Clustering: the scaler and the K-Means model were saved together
clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "20261121_170750_kmeans_farmer_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
clustering_features = clustering_artifact["features"]

# 4. Association rules: precomputed lookup plus the item catalogue
arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "20261121_170352_delivery_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
item_lookup = arm_artifact["item_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


# -----------------------------------------------------------------------
# Input fields each model expects
# -----------------------------------------------------------------------
REGRESSION_NUMERIC = [
    "years_farming_experience", "soil_quality_score", "num_workers_employed",
    "rainfall_mm", "market_distance_km", "farm_size_acres",
    "irrigation_maintenance_cost_kes",
]
REGRESSION_CATEGORICAL = ["crop_type", "county", "irrigation_type", "cooperative_member"]
# The pipeline has its own imputers for numeric columns, so a numeric field
# may be sent as null (for example irrigation_maintenance_cost_kes for a farm
# with no irrigation). The field name must still be present in the request.
# irrigation_type may also be null: the pipeline fills it with "None"
# (no irrigation), exactly as it did during training.
REGRESSION_OPTIONAL = REGRESSION_NUMERIC + ["irrigation_type"]

CLASSIFICATION_NUMERIC = [
    "payment_terms_days", "invoice_amount_kes", "client_relationship_years",
    "prior_late_payments_count", "discount_offered_pct",
    "days_outstanding_avg_historical", "guarantee_value_kes",
    "economic_sector_growth_rate_pct",
]
CLASSIFICATION_CATEGORICAL = [
    "client_industry", "client_credit_rating", "client_company_size",
    "payment_method", "has_dispute_history", "collateral_or_guarantee",
]
# Numeric fields may be null (for example guarantee_value_kes when there is no
# guarantee). The pipeline imputes them. The field name must still be present.
CLASSIFICATION_OPTIONAL = CLASSIFICATION_NUMERIC

CLUSTER_NAMES = {
    0: "Low-access smallholders",
    1: "Established smallholders with good market access",
    2: "Commercial farmers",
    3: "High-value niche farmers",
}


# -----------------------------------------------------------------------
# Basic error handling helpers
# -----------------------------------------------------------------------
def missing_fields(payload, required, optional=()):
    """Return the required fields that are absent or empty in the payload."""
    missing = []
    for field in required:
        if field in optional:
            if field not in payload:
                missing.append(field)
        elif field not in payload or payload[field] is None or str(payload[field]).strip() == "":
            missing.append(field)
    return missing


def build_row(payload, numeric, categorical):
    """Build a one-row DataFrame with numbers as floats (null becomes NaN)."""
    row = {}
    for field in numeric:
        value = payload.get(field)
        row[field] = np.nan if value is None or value == "" else float(value)
    for field in categorical:
        value = payload.get(field)
        row[field] = np.nan if value is None or value == "" else str(value)
    return pd.DataFrame([row])


def bad_request(message):
    return jsonify({"error": message}), 400


# -----------------------------------------------------------------------
# 1. REGRESSION -- predict annual farm income (KES)
# -----------------------------------------------------------------------
# Sample request (cURL, Git Bash):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/farm-income-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"years_farming_experience": 10, "soil_quality_score": 60,
#             "num_workers_employed": 1, "rainfall_mm": 1345,
#             "market_distance_km": 5.7, "farm_size_acres": 7.26,
#             "irrigation_maintenance_cost_kes": 20609.32,
#             "crop_type": "Tea", "county": "Uasin Gishu",
#             "irrigation_type": "Drip", "cooperative_member": "No"}'
#
# irrigation_maintenance_cost_kes may be null for a farm with
# irrigation_type "None". The pipeline's own imputer handles it.
@app.route("/api/v1/models/farm-income-regressor/predictions", methods=["POST"])
def predict_farm_income():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return bad_request("Send a JSON object in the request body.")

    missing = missing_fields(
        payload, REGRESSION_NUMERIC + REGRESSION_CATEGORICAL, REGRESSION_OPTIONAL
    )
    if missing:
        return bad_request("Missing required fields: " + ", ".join(missing))

    try:
        input_row = build_row(payload, REGRESSION_NUMERIC, REGRESSION_CATEGORICAL)
    except (TypeError, ValueError):
        return bad_request("Numeric fields must contain numbers: " + ", ".join(REGRESSION_NUMERIC))

    # The model was trained on log1p(income), so expm1 turns it back into KES.
    predicted_log_income = regression_pipeline.predict(input_row)[0]
    predicted_income_kes = float(np.expm1(predicted_log_income))

    return jsonify({
        "predicted_annual_farm_income_kes": round(predicted_income_kes, 2),
        "model": "Ridge Regressor",
    })


# -----------------------------------------------------------------------
# 2. CLASSIFICATION -- predict invoice payment risk
# -----------------------------------------------------------------------
# Sample request (cURL, Git Bash):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/invoice-risk-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"client_industry": "Technology", "client_credit_rating": "Good",
#             "client_company_size": "Large", "payment_method": "Bank Transfer",
#             "has_dispute_history": "Yes", "collateral_or_guarantee": "No",
#             "payment_terms_days": 45, "invoice_amount_kes": 256337,
#             "client_relationship_years": 1.9, "prior_late_payments_count": 1,
#             "discount_offered_pct": 2.73, "days_outstanding_avg_historical": 22.2,
#             "guarantee_value_kes": null, "economic_sector_growth_rate_pct": 4.87}'
#
# guarantee_value_kes may be null when collateral_or_guarantee is "No".
@app.route("/api/v1/models/invoice-risk-classifier/predictions", methods=["POST"])
def predict_invoice_risk():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return bad_request("Send a JSON object in the request body.")

    missing = missing_fields(
        payload, CLASSIFICATION_NUMERIC + CLASSIFICATION_CATEGORICAL, CLASSIFICATION_OPTIONAL
    )
    if missing:
        return bad_request("Missing required fields: " + ", ".join(missing))

    try:
        input_row = build_row(payload, CLASSIFICATION_NUMERIC, CLASSIFICATION_CATEGORICAL)
    except (TypeError, ValueError):
        return bad_request("Numeric fields must contain numbers: " + ", ".join(CLASSIFICATION_NUMERIC))

    predicted_class = str(classification_pipeline.predict(input_row)[0])
    probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_probabilities = {
        str(label): round(float(p), 4)
        for label, p in zip(classification_pipeline.classes_, probabilities)
    }

    return jsonify({
        "predicted_payment_risk_category": predicted_class,
        "class_probabilities": class_probabilities,
        "model": "Support Vector Classifier (linear kernel, SMOTE)",
    })


# -----------------------------------------------------------------------
# 3. CLUSTERING -- assign a cooperative farmer to a segment
# -----------------------------------------------------------------------
# Sample request (cURL, Git Bash):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/farmer-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"farm_size_acres": 16.29, "annual_yield_tonnes": 35.9,
#             "income_per_acre_kes": 2000, "years_farming_experience": 7,
#             "market_access_score": 51.7}'
@app.route("/api/v1/models/farmer-segmenter/predictions", methods=["POST"])
def predict_farmer_segment():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return bad_request("Send a JSON object in the request body.")

    missing = missing_fields(payload, clustering_features)
    if missing:
        return bad_request("Missing required fields: " + ", ".join(missing))

    try:
        input_row = build_row(payload, clustering_features, [])
    except (TypeError, ValueError):
        return bad_request("All fields must contain numbers: " + ", ".join(clustering_features))

    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "segment_name": CLUSTER_NAMES.get(assigned_cluster, "Unknown"),
        "model": "K-Means (k=4, standardised features)",
    })


# -----------------------------------------------------------------------
# 4. ASSOCIATION RULES -- recommend food items for a cart
# -----------------------------------------------------------------------
# Item IDs: F01 Beef Burger, F02 French Fries, F03 Soda, F04 Chicken Wings,
# F05 Margherita Pizza, F06 Garlic Bread, F07 California Roll Sushi,
# F08 Miso Soup, F09 Green Tea, F10 Chicken Biryani, F11 Raita,
# F12 Naan Bread, F13 Chapati, F14 Chocolate Cake, F15 Ice Cream
#
# Sample request (cURL, Git Bash):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/delivery-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["F01", "F03"]}'
@app.route("/api/v1/models/delivery-recommender/predictions", methods=["POST"])
def recommend_items():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return bad_request("Send a JSON object in the request body.")

    cart_list = payload.get("cart")
    if not isinstance(cart_list, list) or len(cart_list) == 0:
        return bad_request('Missing required field: cart (a list of item IDs, e.g. ["F01", "F03"])')

    cart = {str(item).strip().upper() for item in cart_list}
    unknown = sorted(item for item in cart if item not in item_lookup.index)
    if unknown:
        return bad_request("Unknown item IDs: " + ", ".join(unknown))

    try:
        top_n = int(payload.get("top_n", 3))
    except (TypeError, ValueError):
        return bad_request("top_n must be a whole number.")

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for item, confidence, lift in consequent_list:
                if item not in cart:
                    if item not in candidates or confidence > candidates[item][0]:
                        candidates[item] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "item_id": str(item),
            "item_name": str(item_lookup.loc[item, "item_name"]),
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for item, (confidence, lift) in ranked
    ]

    response = {
        "cart": [
            {"item_id": item, "item_name": str(item_lookup.loc[item, "item_name"])}
            for item in sorted(cart)
        ],
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned and precomputed lookup",
    }
    if not recommendations:
        response["message"] = "No rules match this cart, so there is nothing to recommend."
    return jsonify(response)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server only. Never use debug=True on a public server.
    app.run(debug=True)
