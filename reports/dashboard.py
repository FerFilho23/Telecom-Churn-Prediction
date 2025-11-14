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

st.title("🛜 Telecom Customer Churn")
st.markdown(
    """
    This dashboard summarizes the **churn prediction model** trained on the IBM Telco dataset.
    """
)

st.markdown("## Data")
st.write(df)
