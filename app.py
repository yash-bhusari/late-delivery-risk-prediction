import streamlit as st
import joblib
import os
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Supply Chain Late Delivery Risk Prediction")

st.markdown("""
### Machine Learning-Based Risk Assessment

Predict the possibility of late delivery using machine learning.
""")

st.divider()

# Load trained model and preprocessor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "logistic_regression_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(BASE_DIR, "models", "preprocessor_final.pkl")
)

st.success("Machine Learning model loaded successfully!")

st.subheader("Model Performance")

col1, col2 = st.columns(2)

col1.metric("Test Accuracy", "71.41%")
col2.metric("ROC-AUC Score", "0.7823")
st.subheader("Model Performance Comparison")

comparison_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest (Balanced)",
        "Random Forest (Basic)"
    ],
    "Accuracy": [71.41, 69.89, 55.99],
    "ROC-AUC": [78.23, 76.03, 75.30]
})

st.bar_chart(
    comparison_data.set_index("Model")
)
st.divider()
st.header("📋 Enter Order Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        order_type = st.selectbox(
            "Order Type",
            ["TRANSFER", "DEBIT", "PAYMENT", "CASH"]
        )

        category = st.text_input("Category Name", "Cleats")

        customer_city = st.text_input("Customer City", "New York")

        customer_country = st.text_input("Customer Country", "United States")

        customer_segment = st.selectbox(
            "Customer Segment",
            ["Consumer", "Corporate", "Home Office"]
        )

        customer_state = st.text_input("Customer State", "New York")

        department = st.text_input("Department Name", "Fan Shop")

        latitude = st.number_input("Latitude", value=40.7128)

        longitude = st.number_input("Longitude", value=-74.0060)

        market = st.selectbox(
            "Market",
            ["USCA", "Europe", "LATAM", "Pacific Asia", "Africa"]
        )

        order_city = st.text_input("Order City", "New York")

    with col2:
        order_country = st.text_input("Order Country", "United States")

        discount = st.number_input("Order Item Discount", min_value=0.0)

        discount_rate = st.number_input(
            "Order Item Discount Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )

        quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            value=1
        )

        sales = st.number_input("Sales", min_value=0.0, value=100.0)

        item_total = st.number_input(
            "Order Item Total",
            min_value=0.0,
            value=100.0
        )

        order_region = st.text_input("Order Region", "East")

        order_state = st.text_input("Order State", "New York")

        product_name = st.text_input("Product Name", "Perfect Fitness")

        product_price = st.number_input(
            "Product Price",
            min_value=0.0,
            value=100.0
        )

        shipping_mode = st.selectbox(
            "Shipping Mode",
            ["Standard Class", "Second Class", "First Class", "Same Day"]
        )

    submitted = st.form_submit_button("Predict Delivery Risk")
    if submitted:
        input_data = pd.DataFrame([{
            "Type": order_type,
            "Category Name": category,
            "Customer City": customer_city,
            "Customer Country": customer_country,
            "Customer Segment": customer_segment,
            "Customer State": customer_state,
            "Department Name": department,
            "Latitude": latitude,
            "Longitude": longitude,
            "Market": market,
            "Order City": order_city,
            "Order Country": order_country,
            "Order Item Discount": discount,
            "Order Item Discount Rate": discount_rate,
            "Order Item Quantity": quantity,
            "Sales": sales,
            "Order Item Total": item_total,
            "Order Region": order_region,
            "Order State": order_state,
            "Product Name": product_name,
            "Product Price": product_price,
            "Shipping Mode": shipping_mode
        }])

        input_processed = preprocessor.transform(input_data)

        prediction = model.predict(input_processed)[0]

        probability = model.predict_proba(input_processed)[0][1]

        st.divider()
        st.subheader("Prediction Result")

        st.metric("Late Delivery Probability", f"{probability:.2%}")

        if prediction == 1:
            st.error("⚠️ High Risk: This order is predicted to be delivered late.")
        else:
            st.success("✅ Low Risk: This order is predicted to be delivered on time.")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Late Delivery Risk"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkorange"},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "khaki"},
                    {"range": [70, 100], "color": "lightcoral"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)
        st.divider()

st.header("📌 Project Overview")

st.write("""
This project uses machine learning to identify potential late deliveries
in supply chain operations.

**Objective:**
Predict late-delivery risk using order, product, customer,
geographical and shipping information.

**Selected Model:** Logistic Regression

**Dataset Size:** 180,519 records

**Input Features:** 22

**Test Accuracy:** 71.41%

**ROC-AUC:** 0.7823
""")

st.header("⚠️ Project Limitations")

st.write("""
- Predictions depend on the patterns present in the training dataset.
- The dataset does not contain order-date information, so chronological validation was not possible.
- Predictions are estimates, not guaranteed delivery outcomes.
- Further testing on real operational data is required before deployment.
""")