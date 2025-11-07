# Telecom Customer Churn Prediction

<!-- TODO: Update Badges and the end of the project -->
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![pandas](https://img.shields.io/badge/pandas-1.5+-green?logo=pandas)
![NumPy](https://img.shields.io/badge/numpy-1.23+-orange?logo=numpy)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.7+-blue?logo=matplotlib)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow?logo=scikitlearn)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

## 🧭 Project Overview

This project implements an end-to-end machine learning pipeline to predict customer churn for a fictional telecom company.

It simulates a real-world ML engineering workflow, from data preprocessing and model training to API deployment, following best practices for production-ready pipelines.

### 🎯 Goals

- Build a churn classification model to identify customers likely to cancel their subscription.

- Design a clean, modular codebase that can evolve into production systems.

- Provide an API for real-time inference using FastAPI.

- Ensure reproducibility with environment setup, versioned data, and model artifacts.

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/FerFilho23/telecom-churn-prediction.git
cd telecom-churn-prediction
pip install -r requirements.txt
```
<!-- TODO: Update Repository Structure and the end of the project -->
## 🗂️ Repository Structure

```bash
📁 telecom-churn-prediction/
│
├── 📄 README.md                      # Project documentation
├── 📄 requirements.txt               # Dependencies list
│
├── 📊 data/                          # Datasets
│   ├── raw/                          # Original Telco churn dataset
│   ├── processed/                    # Cleaned / feature-engineered data
│
├── 🧠 models/                        # Trained models and artifacts
│   └── churn_model
│
├── 📓 notebooks/                     # EDA and experiments
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_experiments.ipynb
│
├── 🧩 src/                           # Source code package
│   ├── config.py                     # Config and constants
│   ├── data.py                       # Data loading and validation
│   ├── features.py                   # Feature engineering pipeline
│   ├── train.py                      # Model training script
│   ├── evaluate.py                   # Evaluation and metrics
│   ├── predict.py                    # Batch prediction helper
│   └── api.py                        # FastAPI app for inference
│
└── 🧪 tests/                         # Unit tests
    ├── test_data.py
    ├── test_features.py
    └── test_api.py
```

## 📊 Dataset Description

The dataset used in this project is the [Telco Customer Churn — IBM Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
, from the classic IBM sample data originally distributed with IBM Cognos Analytics 11.1.3+.

It represents a fictional telecommunications company providing home phone and internet services to 7,043 customers in California during the third quarter (Q3).
Each observation corresponds to a unique customer, described by 33 variables capturing personal demographics, account details, service subscriptions, and churn-related metrics.

The target variable is Churn Label, indicating whether a customer left the company during the quarter.

### 📋 Feature Overview

| Category                      | Example Columns                                                                                                                                                    | Description                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Identification & Location** | `CustomerID`, `Country`, `State`, `City`, `Zip Code`, `Latitude`, `Longitude`, `Lat Long`                                                                          | Uniquely identifies each customer and their geographic information.                                             |
| **Demographics**              | `Gender`, `Senior Citizen`, `Partner`, `Dependents`                                                                                                                | Basic customer characteristics such as age group and household composition.                                     |
| **Services Subscribed**       | `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies` | Indicates the telecom services and optional add-ons purchased.                                                  |
| **Account Information**       | `Tenure Months`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charge`, `Total Charges`                                                              | Subscription duration, billing preferences, and total spend.                                                    |
| **Churn Indicators**          | `Churn Label`, `Churn Value`, `Churn Score`, `Churn Reason`, `Churn Category`                                                                                      | Binary label (Yes/No), numeric label (0/1), IBM SPSS predictive score (0–100), and specific churn explanations. |
| **Customer Value Metrics**    | `CLTV`                                                                                                                                                             | Predicted **Customer Lifetime Value**, indicating the customer’s expected revenue contribution.                 |

## 💻 Modeling Approach

1. Exploratory Data Analysis (EDA)
    - Load and clean data.

    - Visualize churn rates across demographic and service categories.

2. Feature Engineering
    - Build preprocessing pipeline with ColumnTransformer.

    - Apply scaling, one-hot encoding, etc.

3. Model Training
    - Train and tune models (LogisticRegression, XGBoost, RandomForest).

    - Evaluate using ROC AUC, Precision-Recall AUC, and Confusion Matrix.

4. Pipeline Export
    - Save preprocessing + model as a single sklearn.Pipeline.

5. API Deployment
    - Serve model predictions via FastAPI endpoint /predict.

    - Validate inputs with Pydantic schemas.

    - Deploy API to GCP cloud.

## 🚀 Future Work

- Integrate experiment tracking with MLflow or Weights & Biases.

- Add Dockerfile and deploy API to Google Cloud Run or Render.

- Implement data validation tests using pydantic or pandera.

- Add monitoring metrics (e.g., data drift, model performance over time).
