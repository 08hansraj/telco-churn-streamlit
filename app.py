import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("churn_xgb_smote_pipeline.pkl")


# -----------------------------
# Tenure binning function
# -----------------------------
def tenure_to_bin(tenure):
    if tenure <= 12:
        return "0-12"
    elif tenure <= 24:
        return "13-24"
    elif tenure <= 36:
        return "25-36"
    elif tenure <= 48:
        return "37-48"
    elif tenure <= 60:
        return "49-60"
    else:
        return "61-72"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📉 Customer Churn Prediction (XGBoost)")
st.write("Fill the customer details below and predict whether the customer is likely to churn.")


with st.form("churn_form"):

    st.subheader("Customer Profile")

    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.subheader("Services")

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.subheader("Contract & Billing")

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    st.subheader("Charges")

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=10.0)

    submit = st.form_submit_button("Predict")


# -----------------------------
# Prediction logic
# -----------------------------
if submit:

    tenure_bin = tenure_to_bin(tenure)

    input_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "tenure_bin": tenure_bin
    }

    input_df = pd.DataFrame([input_dict])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.divider()

    if pred == 1:
        st.error(f"Prediction: CHURN (Yes)")
    else:
        st.success(f"Prediction: NOT CHURN (No)")

    st.info(f"Churn Probability (Class 1): **{prob:.4f}**")

    # Optional: show the final input row
    with st.expander("Show input data"):
        st.dataframe(input_df)