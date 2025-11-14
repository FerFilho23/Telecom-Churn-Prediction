# Telecom Customer Churn Prediction

<!-- TODO: Update Badges and the end of the project -->
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-green?logo=pandas)
![NumPy](https://img.shields.io/badge/Numpy-1.23+-orange?logo=numpy)
![Matplotlib](https://img.shields.io/badge/plotly-6.4+-blue?logo=plotly)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow?logo=scikitlearn)
![gcp](https://img.shields.io/badge/Google--Cloud-blue?logo=google-cloud&logoColor=white)
![streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/version-v1.0-green)

## 🧭 Project Overview

This project implements an end-to-end machine learning pipeline to predict customer churn for a fictional telecom company.

It simulates a real-world ML engineering workflow, from data preprocessing and model training to API deployment, following best practices for production-ready pipelines.

### 🎯 Goals

- Build a churn classification model to identify customers likely to cancel their subscription.

- Design a clean, modular codebase that can evolve into production systems.

- Provide an API for real-time inference using FastAPI.

- Ensure reproducibility with environment setup, versioned data, and model artifacts.

## 🚀 Live API (Google Cloud Run)

This ML model is deployed and running serverlessly on Google Cloud Run:

https://telecom-churn-prediction-91434658183.us-central1.run.app/docs


### ☁️ GCP Architecture Diagram
GitHub Repo → Cloud Build Trigger → Build Container Image → Deploy to Cloud Run → Public FastAPI Endpoint


## ⚙️ Local Development

Create a [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) env:

```bash
conda create -n telecom-churn python=3.11
conda activate telecom-churn
```

Clone the repository and install dependencies:

```bash
git clone https://github.com/FerFilho23/telecom-churn-prediction.git
cd telecom-churn-prediction
pip install -r requirements.txt
```

Run FastAPI locally

```bash
fastapi dev src/app.py
```

Read the API documentation at: http://127.0.0.1:8000/docs

Docker build

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Run Streamlit

```bash
streamlit run reports/dashboard.py
```

## 🗂️ Repository Structure

```bash
📁 telecom-churn-prediction/
│
├── 📄 README.md                      # Project documentation
├── 📄 requirements.txt               # Dependencies
├── 📄 Dockerfile                     # Containerized deployment
│
├── 📊 data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Cleaned data used for modeling
│
├── 🧠 models/
│   └─ v1_thrs_0.35.joblib           # Final tuned model
│
├── 📓 notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_model_training.ipynb       # Model training and tuning
│
├── ⚙️ app/
│   └── app.py                        # FastAPI prediction service
│
└── 📊 reports/
    └── streamlit_dashboard.py        # Streamlit dashboard

```

## 📊 Dataset Description

**Dataset:** [Telco Customer Churn — IBM Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

**Size:** 7,043 customers • 33 features
Context: Fictional telecom provider in California (Q3 snapshot)

**Target variable:**
Churn Value — 1 if the customer churned, 0 otherwise.


## 🧠 Modeling Approach

1. Exploratory Data Analysis (EDA)
    - Load and clean data.

    - Explore and visualize churn distribution.

2. Feature Engineering
    - Applied **Mutual Information (MI)** as a feature selection strategy.

    - Retained top 15 relevant features.

    - Build preprocessing pipeline with ColumnTransformer.

    - Apply scaling, one-hot encoding, etc.

3. Model Training
    - Train and tune models (LogisticRegression, XGBoost, RandomForest, Gradient Boosting).

    - Evaluate using ROC AUC, Precision-Recall AUC, and F1.

    - Tuned the decision threshold for F1 Optimization. 

4. Pipeline Export
    - Save preprocessing + model as a single sklearn.Pipeline.

5. API Deployment
    - Serve model predictions via FastAPI endpoint /predict.

    - Validate inputs with Pydantic schemas.

    - Deploy API to GCP cloud through Cloud Build.

## 🚀 Future Work

- Improve model performance

- Experiment tracking with MLflow or Weights & Biases

- Automated retraining pipeline

- Monitoring drift in data and model performance
