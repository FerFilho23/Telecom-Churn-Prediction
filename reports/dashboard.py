from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio

# ----------------- CONFIG -----------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "v1_thrs_0.35.joblib"
THRESHOLD = 0.35
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_customer_churn.xlsx"


# ----------------- LOAD ARTIFACTS -----------------
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = THRESHOLD
    return model, threshold


def load_data():
    if DATA_PATH.exists():
        df = pd.read_excel(DATA_PATH)
        return df
    return None


pipeline, default_threshold = load_model()
df = load_data()

st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="🛜",
    layout="wide",
)

# Define the input features for the model
feature_names = [
    "senior_citizen",
    "dependents",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "tenure_months",
    "monthly_charges",
    "total_charges",
]

default_values = {
    "senior_citizen": "Yes",
    "dependents": "Yes",
    "internet_service": "Fiber optic",
    "online_security": "Yes",
    "online_backup": "Yes",
    "device_protection": "Yes",
    "tech_support": "Yes",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Credit card (automatic)",
    "tenure_months": 18,
    "monthly_charges": 99.65,
    "total_charges": 1820.50,
}

st.sidebar.header("User Inputs")

st.title("🛜 Telecom Customer Churn")
st.markdown(
    """
    This dashboard summarizes the **churn prediction model** trained on the [Telco Customer Churn — IBM Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset). 
    """
)

# Page Layout
left_col, right_col = st.columns(2)

# Left Page: Feature Importance
with left_col:
    st.header("Feature Importance")

# Right Page: Prediction
with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        st.markdown(f"### Output: ")
