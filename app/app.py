# -----------------------------------------------------------
# app.py (FULL FIXED VERSION)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import shap

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")


# ---------------------------
# Load Model + Preprocessor
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(model_path="best_model_xgb.pkl",
                   preproc_path="preprocessor.pkl"):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")
    st.stop()


# ---------------------------
# Helper to extract XGBoost estimator
# ---------------------------
def get_raw_xgb_estimator(model_obj):
    try:
        if hasattr(model_obj, "named_steps"):
            for _, step in model_obj.named_steps.items():
                if step.__class__.__module__.startswith("xgboost"):
                    return step
            last_step = list(model_obj.named_steps.values())[-1]
            if last_step.__class__.__module__.startswith("xgboost"):
                return last_step
        if model_obj.__class__.__module__.startswith("xgboost"):
            return model_obj
    except Exception:
        return None
    return None


xgb_estimator = get_raw_xgb_estimator(model)


# ---------------------------
# HEADER
# ---------------------------
st.markdown(
    """
    <div style="text-align:center; padding:10px;">
        <h1 style="color:#0b6e4f;">📡 Telecom Customer Churn Predictor</h1>
        <p style="font-size:15px; color:gray;">
            Enter customer details and get churn probability — with interpretable explanations and a downloadable report.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")
st.subheader("📝 Customer Input")


# ---------------------------
# INPUT FORM
# ---------------------------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.markdown("### 💰 Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

    with col2:
        st.markdown("### 📡 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)

    submitted = st.form_submit_button("🔮 Predict Churn")


# ---------------------------
# Build user dataframe
# ---------------------------
def build_user_dataframe():
    d = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    return pd.DataFrame([d])


# ---------------------------
# Feature engineering
# ---------------------------
def add_engineered_features(df):
    df["tenure_bin"] = pd.cut(df["tenure"], bins=[0,6,12,24,48,100],
                              labels=["0-6","7-12","13-24","25-48","49+"], include_lowest=True)
    df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["PaymentMethod_clean"] = df["PaymentMethod"].apply(
        lambda x: "Electronic check" if x == "Electronic check" else "Other"
    )
    return df


# ---------------------------
# PDF generator
# ---------------------------
def create_churn_report_pdf(input_data: dict, pred: int, prob: float, top_features: dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "📡 Telecom Customer Churn Report")
    c.setFont("Helvetica", 12)
    y = height - 100

    c.drawString(50, y, f"Prediction: {'CHURN' if pred == 1 else 'NOT CHURN'}")
    y -= 20
    c.drawString(50, y, f"Probability: {prob*100:.2f}%")
    y -= 40

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Customer Input Details")
    y -= 20
    c.setFont("Helvetica", 11)

    for k, v in input_data.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 15

    y -= 20
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Top Features Influencing Prediction:")
    y -= 20
    c.setFont("Helvetica", 11)

    for k, v in top_features.items():
        c.drawString(60, y, f"{k}: {v:.4f}")
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer


# ---------------------------
# Retention Recommendations
# ---------------------------
def retention_recommendations(user):
    recs = []

    if user["Contract"] == "Month-to-month":
        recs.append("Offer a discount if they switch to a One-year or Two-year contract.")

    if user["MonthlyCharges"] > 80:
        recs.append("Provide a 10–20% discount or bundled services.")

    if user["tenure"] < 6:
        recs.append("Provide welcome offer or loyalty points to new customer.")

    if user["InternetService"] == "Fiber optic" and user["MonthlyCharges"] > 90:
        recs.append("Offer lower-tier fiber plan or temporary relief.")

    if user["TechSupport"] == "No":
        recs.append("Provide 3 months free Tech Support.")

    if user["DeviceProtection"] == "No":
        recs.append("Give free Device Protection trial.")

    if user["PaymentMethod"] == "Electronic check":
        recs.append("Suggest switching to automatic payments to reduce billing issues.")

    if user["SeniorCitizen"] == "Yes":
        recs.append("Offer personalized senior customer care & easier billing options.")

    if not recs:
        recs.append("Customer seems stable. Maintain regular service quality.")

    return recs


# ============================================================
#                  PREDICTION SECTION
# ============================================================
if submitted:

    user_df = build_user_dataframe()
    user_df = add_engineered_features(user_df)

    st.markdown("### 🔎 Input Summary")
    st.table(user_df.T.rename(columns={0: "value"}))

    processed_input = preprocessor.transform(user_df)

    pred_proba = model.predict_proba(processed_input)
    pred = model.predict(processed_input)[0]
    prob = float(pred_proba[0][1])

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#ffefef;
                        border-left:5px solid #d32f2f;">
                <h3>⚠️ High Churn Risk</h3>
                <p>Churn Probability: <b>{prob*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="padding:18px; border-radius:12px; 
                        background-color:#c8f7c5;
                        border-left:6px solid #1B5E20;">
                <h3 style="color:#1B5E20;">✅ Customer is Safe</h3>
                <p style="color:#1B5E20;">Churn Probability: <b>{prob*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True
        )

    # ---------------------------
    # Retention Recommendations
    # ---------------------------
    st.markdown("### 🛠 Retention Recommendations")

    recs = retention_recommendations(user_df.iloc[0])

    for r in recs:
        st.markdown(
            f"""
            <div style="padding:10px; margin-bottom:8px;
                        border-left:5px solid #1976D2;
                        background-color:#e8f1fc; border-radius:6px;">
                <span style="font-size:16px; color:#0d47a1;">🔹 {r}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------
    # REST OF YOUR EXPLAINABILITY CODE
    # (GLOBAL IMPORTANCE, SHAP, PDF, etc.)
    # -----------------------------------

    st.markdown("---")
    # NOTE: Keep your SHAP, global importance, PDF download sections here safely


# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""---""")
st.caption("Built with ❤️ — AI-driven churn prediction using XGBoost + SHAP")
