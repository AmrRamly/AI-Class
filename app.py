import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "medical_insurance_model.pkl"
FEATURES_PATH = "model_feature_columns.csv"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} was not found.")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_columns():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"{FEATURES_PATH} was not found.")
    feature_df = pd.read_csv(FEATURES_PATH)
    if "Feature" not in feature_df.columns:
        raise ValueError("model_feature_columns.csv must contain a 'Feature' column.")
    return feature_df["Feature"].tolist()

try:
    model = load_model()
    model_columns = load_feature_columns()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

def get_bmi_status(value):
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal"
    if value < 30:
        return "Overweight"
    return "Obese"

def get_cost_category(value):
    if value < 5000:
        return "Low"
    if value < 15000:
        return "Moderate"
    if value < 30000:
        return "High"
    return "Very High"

st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f4f7fb 0%, #eef2f7 100%);
    border-right: 1px solid #e1e6ed;
}
.hero-banner {
    padding: 28px 32px;
    border-radius: 18px;
    background: linear-gradient(135deg, #f7fbff 0%, #edf4ff 100%);
    border: 1px solid #dbe7f5;
    box-shadow: 0 8px 22px rgba(34, 60, 100, 0.08);
    text-align: center;
    margin-bottom: 28px;
}
.main-title {
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    line-height: 1.15;
    margin: 0;
    color: #1f2937;
}
.subtitle {
    font-size: 17px;
    color: #5f6b7a;
    margin-top: 10px;
    margin-bottom: 0;
}
.prediction-card {
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(135deg, #eef8f4 0%, #e4f3ed 100%);
    border: 1px solid #c7e4d8;
    box-shadow: 0 8px 22px rgba(30, 100, 75, 0.10);
    margin-top: 20px;
    margin-bottom: 20px;
}
.prediction-label {
    font-size: 18px;
    color: #40524b;
    margin-bottom: 5px;
}
.prediction-value {
    font-size: clamp(38px, 5vw, 56px);
    line-height: 1.1;
    font-weight: 800;
    color: #183f32;
    margin: 4px 0 10px;
}
.prediction-note {
    font-size: 13px;
    color: #66756f;
    margin: 0;
}
.mini-card {
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #e2e7ee;
    background: #fafbfd;
    text-align: center;
    min-height: 105px;
}
.mini-label {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 6px;
}
.mini-value {
    font-size: 20px;
    font-weight: 700;
    color: #263445;
}
div.stButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
    border: 0;
}
.footer-box {
    margin-top: 34px;
    padding: 20px 24px;
    border-top: 1px solid #dfe5ec;
    text-align: center;
    color: #5d6875;
    font-size: 14px;
    line-height: 1.7;
}
.footer-name {
    color: #263445;
    font-weight: 700;
    font-size: 15px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <div class="main-title">🏥 Medical Insurance Cost Prediction</div>
    <div class="subtitle">
        AI-based regression dashboard for estimating medical insurance charges
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About the System")
    st.write(
        "This dashboard estimates medical insurance charges using a trained regression model."
    )
    st.divider()
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² Score", "0.958")
        st.metric("MAE", "$1,710")
    with c2:
        st.metric("RMSE", "$2,076")
        st.metric("Model", "XGBoost")
    st.divider()
    st.subheader("Model Inputs")
    st.write("Age, BMI, number of children, sex, smoking status and region.")
    with st.expander("View Encoded Features"):
        st.code("\n".join(model_columns))
    st.warning(
        "For educational demonstration only. This prediction is not an official insurance quotation."
    )

st.subheader("Applicant Information")
left, middle, right = st.columns(3)

with left:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", ["Female", "Male"])

with middle:
    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        format="%.1f",
    )
    children = st.number_input(
        "Number of Children", min_value=0, max_value=10, value=0, step=1
    )

with right:
    smoker = st.selectbox("Smoking Status", ["No", "Yes"])
    region = st.selectbox(
        "Region", ["Northeast", "Northwest", "Southeast", "Southwest"]
    )

with st.expander("View Feature Encoding"):
    encoding_table = pd.DataFrame({
        "Feature": ["Sex", "Sex", "Smoker", "Smoker", "Region", "Region", "Region", "Region"],
        "Original Value": ["Female", "Male", "No", "Yes", "Northeast", "Northwest", "Southeast", "Southwest"],
        "Encoded Value": ["0", "1", "0", "1", "(0, 0, 0)", "(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)"],
    })
    st.dataframe(encoding_table, use_container_width=True, hide_index=True)

if st.button("Predict Insurance Cost", type="primary", use_container_width=True):
    input_data = pd.DataFrame([[0] * len(model_columns)], columns=model_columns)

    if "Age" in input_data.columns:
        input_data.loc[0, "Age"] = age
    if "BMI" in input_data.columns:
        input_data.loc[0, "BMI"] = bmi
    if "Children" in input_data.columns:
        input_data.loc[0, "Children"] = children
    if "Sex_male" in input_data.columns:
        input_data.loc[0, "Sex_male"] = 1 if sex == "Male" else 0
    if "Smoker_yes" in input_data.columns:
        input_data.loc[0, "Smoker_yes"] = 1 if smoker == "Yes" else 0

    region_mapping = {
        "Northeast": None,
        "Northwest": "Region_northwest",
        "Southeast": "Region_southeast",
        "Southwest": "Region_southwest",
    }
    selected_region_column = region_mapping[region]

    if selected_region_column and selected_region_column in input_data.columns:
        input_data.loc[0, selected_region_column] = 1

    try:
        predicted_charge = max(float(model.predict(input_data)[0]), 0.0)
        bmi_status = get_bmi_status(bmi)
        cost_category = get_cost_category(predicted_charge)

        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated Medical Insurance Cost</div>
            <div class="prediction-value">${predicted_charge:,.2f}</div>
            <p class="prediction-note">
                Prediction generated using the trained regression model
            </p>
        </div>
        """, unsafe_allow_html=True)

        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown(
                f'<div class="mini-card"><div class="mini-label">Cost Category</div>'
                f'<div class="mini-value">{cost_category}</div></div>',
                unsafe_allow_html=True,
            )
        with i2:
            st.markdown(
                f'<div class="mini-card"><div class="mini-label">BMI Category</div>'
                f'<div class="mini-value">{bmi_status}</div></div>',
                unsafe_allow_html=True,
            )
        with i3:
            st.markdown(
                '<div class="mini-card"><div class="mini-label">Model Used</div>'
                '<div class="mini-value">Linear Regression</div></div>',
                unsafe_allow_html=True,
            )

        st.subheader("Applicant Summary")
        summary_table = pd.DataFrame({
            "Feature": [
                "Age", "Sex", "BMI", "BMI Category",
                "Children", "Smoker", "Region", "Prediction Time"
            ],
            "Value": [
                age, sex, f"{bmi:.1f}", bmi_status,
                children, smoker, region,
                datetime.now().strftime("%d %b %Y, %I:%M %p")
            ],
        })
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        with st.expander("View Encoded Model Input"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")

st.markdown("""
<div class="footer-box">
    <div class="footer-name">Developed by Ir. Dr. Marni Azira Markom</div>
    Faculty of Intelligent Computing, Universiti Malaysia Perlis (UniMAP)<br>
    AI Regression Workshop 2026
</div>
""", unsafe_allow_html=True)
