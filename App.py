#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib

# Load the saved model and encoders
model = joblib.load("model.pkl")
oe = joblib.load("ordinal_encoder.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Stock Risk Prediction", page_icon="📦", layout="centered")
st.title("📦 Inventory Stock Risk Prediction")
st.write("Predict whether an inventory item is at risk of overstock or stockout.")
st.subheader("Enter Inventory Details")

# Numerical inputs
current_stock = st.number_input("Current Stock", min_value=0.0, value=500.0)
avg_daily_demand = st.number_input("Average Daily Demand", min_value=0.0, value=40.0)
lead_time_days = st.number_input("Lead Time (Days)", min_value=0.0, value=6.0)
reorder_point = st.number_input("Reorder Point", min_value=0.0, value=350.0)
sales_last_30_days = st.number_input("Sales in Last 30 Days", min_value=0.0, value=1500.0)
stock_turnover_ratio = st.number_input("Stock Turnover Ratio", min_value=0.0, value=4.0)
forecast_error = st.number_input("Forecast Error", value=0.0)

# Categorical inputs
season = st.selectbox("Season", ["off-peak", "peak"])
item_category = st.selectbox("Item Category", ["essential", "non-essential"])
supplier_reliability = st.selectbox("Supplier Reliability", ["high", "low", "medium"])

if st.button("Predict Stock Status"):
    try:
        # Keep categorical columns in the same order used during training
        categorical_data = pd.DataFrame([{"season":season,"item_category":item_category,"supplier_reliability":supplier_reliability}])

        # Apply the encoder fitted during training
        encoded_categories = oe.transform(categorical_data)

        # Build the input in the exact training feature order
        input_data = pd.DataFrame([{
            "current_stock": current_stock,
            "avg_daily_demand": avg_daily_demand,
            "lead_time_days": lead_time_days,
            "reorder_point": reorder_point,
            "sales_last_30_days": sales_last_30_days,
            "stock_turnover_ratio": stock_turnover_ratio,
            "forecast_error": forecast_error,
            "season": encoded_categories[0][0],
            "item_category": encoded_categories[0][1],
            "supplier_reliability": encoded_categories[0][2]
        }])

        # Predict using the complete fitted pipeline
        prediction = model.predict(input_data)[0]
        predicted_status = le.inverse_transform([prediction])[0]

        st.subheader("Prediction Result")

        if predicted_status == "stockout":
            st.error("⚠️ Stockout Risk Predicted")
        else:
            st.success("✅ Overstock Predicted")

        st.write("Predicted Stock Status:", predicted_status)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

