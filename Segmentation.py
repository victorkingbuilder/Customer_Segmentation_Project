import streamlit as st
import pandas as pd
import joblib

try:
    import numpy as np
except Exception as e:
    np = None
    st.title("Customer Segmentation Prediction App")
    st.error("NumPy failed to import: {}".format(e))
    st.info(
        "To fix this, install a compatible NumPy in your Python environment:\n"
        "python -m pip install --upgrade --force-reinstall \"numpy>=1.24.3,<2.0\""
    )
    st.stop()

kmeans = joblib.load("customer_segmentation_kmeans_model.pkl")
scaler = joblib.load("customer_segmentation_scaler.pkl")

st.title("Customer Segmentation Prediction App")
st.write("Enter customer details to predict their segment:")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income (k$)", min_value=0.0, max_value=200000.0, value=50000.0)
total_spending = st.number_input("Total Spending (sum of purchases)", min_value=0.0, max_value=5000.0, value=1000.0)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=100, value=10)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

if st.button("Predict Segment"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Total_Spending": total_spending,
        "NumWebPurchases": num_web_purchases,
        "NumStorePurchases": num_store_purchases,
        "NumWebVisitsMonth": num_web_visits,
        "Recency": recency
    }])

    # Ensure exact training order
    input_data = input_data[scaler.feature_names_in_]

    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Segment: {cluster}")