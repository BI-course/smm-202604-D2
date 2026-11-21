import gradio as gr
import joblib
import numpy as np
import pandas as pd
import spaces

# -----------------------------------------------------------------------
# Hugging Face's free tier provisions new Gradio Spaces on ZeroGPU hardware.
# None of our four models needs a GPU, but some free-tier Spaces fail to
# start unless at least one function is decorated with @spaces.GPU. This
# function does nothing GPU-related; it only satisfies that platform check.
# -----------------------------------------------------------------------
@spaces.GPU
def _warm_zerogpu():
    return None

_warm_zerogpu()

# -----------------------------------------------------------------------
# Load all four persisted artifacts ONCE, at import time (same principle
# as api.py). The .joblib files sit next to app.py, so there is no model/ prefix.
# -----------------------------------------------------------------------
regression_pipeline = joblib.load("20261121_170647_ridge_regressor_for_farm_income.joblib")
classification_pipeline = joblib.load("20261121_171040_svc_classifier_for_invoice_risk.joblib")

clustering_artifact = joblib.load("20261121_170750_kmeans_farmer_segmentation.joblib")
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load("20261121_170352_delivery_recommendation_rules.joblib")
recommendation_lookup = arm_artifact["recommendation_lookup"]
item_lookup = arm_artifact["item_lookup"]

# Item IDs and names (section 4 of the "Who Does What" document)
ITEMS = {
    "F01": "Beef Burger", "F02": "French Fries", "F03": "Soda",
    "F04": "Chicken Wings", "F05": "Margherita Pizza", "F06": "Garlic Bread",
    "F07": "California Roll Sushi", "F08": "Miso Soup", "F09": "Green Tea",
    "F10": "Chicken Biryani", "F11": "Raita", "F12": "Naan Bread",
    "F13": "Chapati", "F14": "Chocolate Cake", "F15": "Ice Cream",
}
ITEM_CHOICES = [f"{item_id} - {name}" for item_id, name in ITEMS.items()]

SEGMENT_NAMES = {
    0: "Low-access smallholders",
    1: "Established smallholders with good market access",
    2: "Commercial farmers",
    3: "High-value niche farmers",
}

MISSING_MESSAGE = "Please fill in all fields"


def _is_blank(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def _optional_number(value):
    """Optional numeric fields: a blank box means 'not applicable' (missing)."""
    return np.nan if value is None else value


def _build_row(payload, numeric_fields):
    """Gradio sends whole numbers as int; the fitted imputers expect float64,
    so every numeric column is cast to float before predicting."""
    row = pd.DataFrame([payload])
    for column in numeric_fields:
        row[column] = row[column].astype("float64")
    return row


REGRESSION_NUMERIC = ["years_farming_experience", "soil_quality_score",
                      "num_workers_employed", "rainfall_mm", "market_distance_km",
                      "farm_size_acres", "irrigation_maintenance_cost_kes"]
CLASSIFICATION_NUMERIC = ["payment_terms_days", "invoice_amount_kes",
                          "client_relationship_years", "prior_late_payments_count",
                          "discount_offered_pct", "days_outstanding_avg_historical",
                          "guarantee_value_kes", "economic_sector_growth_rate_pct"]


# -----------------------------------------------------------------------
# 1. Regression -- predict annual farm income (KES)
# -----------------------------------------------------------------------
def predict_farm_income(years_farming_experience, soil_quality_score,
                        num_workers_employed, rainfall_mm, market_distance_km,
                        farm_size_acres, irrigation_maintenance_cost_kes,
                        crop_type, county, irrigation_type, cooperative_member):
    required = [years_farming_experience, num_workers_employed, rainfall_mm,
                market_distance_km, farm_size_acres, crop_type, county,
                irrigation_type, cooperative_member]
    if any(_is_blank(v) for v in required):
        return MISSING_MESSAGE

    payload = {
        "years_farming_experience": years_farming_experience,
        "soil_quality_score": _optional_number(soil_quality_score),
        "num_workers_employed": num_workers_employed,
        "rainfall_mm": rainfall_mm,
        "market_distance_km": market_distance_km,
        "farm_size_acres": farm_size_acres,
        "irrigation_maintenance_cost_kes": _optional_number(irrigation_maintenance_cost_kes),
        "crop_type": crop_type,
        "county": county,
        "irrigation_type": irrigation_type,
        "cooperative_member": cooperative_member,
    }
    input_row = _build_row(payload, REGRESSION_NUMERIC)
    predicted_log_income = regression_pipeline.predict(input_row)[0]
    predicted_income_kes = float(np.expm1(predicted_log_income))
    return f"KES {predicted_income_kes:,.2f} per year"


# -----------------------------------------------------------------------
# 2. Classification -- predict invoice payment risk category
# -----------------------------------------------------------------------
def predict_invoice_risk(client_industry, client_credit_rating, client_company_size,
                         payment_method, has_dispute_history, collateral_or_guarantee,
                         payment_terms_days, invoice_amount_kes,
                         client_relationship_years, prior_late_payments_count,
                         discount_offered_pct, days_outstanding_avg_historical,
                         guarantee_value_kes, economic_sector_growth_rate_pct):
    required = [client_industry, client_credit_rating, client_company_size,
                payment_method, has_dispute_history, collateral_or_guarantee,
                payment_terms_days, invoice_amount_kes, client_relationship_years,
                prior_late_payments_count, discount_offered_pct,
                days_outstanding_avg_historical, economic_sector_growth_rate_pct]
    if any(_is_blank(v) for v in required):
        return MISSING_MESSAGE

    payload = {
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
        "guarantee_value_kes": _optional_number(guarantee_value_kes),
        "economic_sector_growth_rate_pct": economic_sector_growth_rate_pct,
    }
    input_row = _build_row(payload, CLASSIFICATION_NUMERIC)
    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.classes_
    probability_lines = "\n".join(
        f"  {label}: {prob:.1%}" for label, prob in zip(class_labels, predicted_probabilities)
    )
    return f"Predicted payment risk: {predicted_class}\n\nClass probabilities:\n{probability_lines}"


# -----------------------------------------------------------------------
# 3. Clustering -- assign a farmer to an existing segment
# -----------------------------------------------------------------------
def predict_farmer_segment(farm_size_acres, annual_yield_tonnes, income_per_acre_kes,
                           years_farming_experience, market_access_score):
    values = [farm_size_acres, annual_yield_tonnes, income_per_acre_kes,
              years_farming_experience, market_access_score]
    if any(_is_blank(v) for v in values):
        return MISSING_MESSAGE

    input_row = pd.DataFrame([{
        "farm_size_acres": farm_size_acres,
        "annual_yield_tonnes": annual_yield_tonnes,
        "income_per_acre_kes": income_per_acre_kes,
        "years_farming_experience": years_farming_experience,
        "market_access_score": market_access_score,
    }])
    input_row = input_row.astype("float64")
    scaled_input = clustering_scaler.transform(input_row)
    cluster = int(clustering_kmeans.predict(scaled_input)[0])
    segment_name = SEGMENT_NAMES.get(cluster, "Unknown segment")
    return f"Assigned cluster: {cluster}\nSegment: {segment_name}"


# -----------------------------------------------------------------------
# 4. Association rules -- recommend delivery items for a cart
# -----------------------------------------------------------------------
def _item_name(item_id):
    try:
        return item_lookup.loc[item_id, "item_name"]
    except Exception:
        return ITEMS.get(item_id, item_id)


def recommend_items(selected_items, top_n):
    if not selected_items:
        return MISSING_MESSAGE
    cart = {choice.split(" - ")[0].strip() for choice in selected_items}
    top_n = int(top_n) if top_n else 3

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for item_id, confidence, lift in consequent_list:
                if item_id not in cart:
                    if item_id not in candidates or confidence > candidates[item_id][0]:
                        candidates[item_id] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    if not ranked:
        return "No recommendations for this cart. Try a different combination of items."

    lines = [
        f"{_item_name(item_id)} ({item_id})  --  confidence {confidence:.1%}, lift {lift:.2f}"
        for item_id, (confidence, lift) in ranked
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------
# One gr.Blocks() app, four gr.Tab()s -- one shared Space URL
# -----------------------------------------------------------------------
with gr.Blocks(title="Group D2 -- Model Demos") as demo:
    gr.Markdown("# Group D2 -- Model Demos\nSwitch tabs to try a different model.")

    with gr.Tab("Farm Income Regressor"):
        gr.Interface(
            fn=predict_farm_income,
            inputs=[
                gr.Number(label="Years of Farming Experience", value=10),
                gr.Number(label="Soil Quality Score (leave blank if unknown)", value=None),
                gr.Number(label="Number of Workers Employed", value=1),
                gr.Number(label="Rainfall (mm)", value=1345),
                gr.Number(label="Market Distance (km)", value=5.7),
                gr.Number(label="Farm Size (acres)", value=7.26),
                gr.Number(label="Irrigation Maintenance Cost (KES, blank if no irrigation)",
                          value=20609.32),
                gr.Dropdown(label="Crop Type",
                            choices=["Coffee", "Dairy", "Horticulture", "Maize", "Tea"],
                            value="Tea"),
                gr.Dropdown(label="County",
                            choices=["Kericho", "Kiambu", "Meru", "Nakuru", "Other", "Uasin Gishu"],
                            value="Uasin Gishu"),
                gr.Dropdown(label="Irrigation Type",
                            choices=["Automated", "Drip", "Manual", "None"],
                            value="Drip"),
                gr.Dropdown(label="Cooperative Member",
                            choices=["No", "Yes"], value="No"),
            ],
            outputs=gr.Textbox(label="Predicted Annual Farm Income"),
        )

    with gr.Tab("Invoice Risk Classifier"):
        gr.Interface(
            fn=predict_invoice_risk,
            inputs=[
                gr.Dropdown(label="Client Industry",
                            choices=["Construction", "Healthcare", "Manufacturing",
                                     "Retail", "Technology"],
                            value="Technology"),
                gr.Dropdown(label="Client Credit Rating",
                            choices=["Poor", "Fair", "Good", "Excellent"], value="Good"),
                gr.Dropdown(label="Client Company Size",
                            choices=["Small", "Medium", "Large", "Enterprise"], value="Large"),
                gr.Dropdown(label="Payment Method",
                            choices=["Bank Transfer", "Cheque", "Letter of Credit",
                                     "Mobile Money"],
                            value="Bank Transfer"),
                gr.Dropdown(label="Has Dispute History",
                            choices=["No", "Yes"], value="Yes"),
                gr.Dropdown(label="Collateral or Guarantee",
                            choices=["No", "Yes"], value="No"),
                gr.Number(label="Payment Terms (days)", value=45),
                gr.Number(label="Invoice Amount (KES)", value=256337),
                gr.Number(label="Client Relationship (years)", value=1.9),
                gr.Number(label="Prior Late Payments (count)", value=1),
                gr.Number(label="Discount Offered (%)", value=2.73),
                gr.Number(label="Average Historical Days Outstanding", value=22.2),
                gr.Number(label="Guarantee Value (KES, blank if no guarantee)", value=None),
                gr.Number(label="Economic Sector Growth Rate (%)", value=4.87),
            ],
            outputs=gr.Textbox(label="Predicted Payment Risk"),
        )

    with gr.Tab("Farmer Segmenter"):
        gr.Interface(
            fn=predict_farmer_segment,
            inputs=[
                gr.Number(label="Farm Size (acres)", value=16.29),
                gr.Number(label="Annual Yield (tonnes)", value=35.9),
                gr.Number(label="Income per Acre (KES)", value=2000),
                gr.Number(label="Years of Farming Experience", value=7),
                gr.Number(label="Market Access Score", value=51.7),
            ],
            outputs=gr.Textbox(label="Farmer Segment"),
        )

    with gr.Tab("Delivery Recommender"):
        gr.Interface(
            fn=recommend_items,
            inputs=[
                gr.Dropdown(label="Items in the cart", choices=ITEM_CHOICES,
                            multiselect=True,
                            value=["F01 - Beef Burger", "F03 - Soda"]),
                gr.Number(label="Top N Recommendations", value=3),
            ],
            outputs=gr.Textbox(label="Recommendations"),
        )

if __name__ == "__main__":
    demo.launch()