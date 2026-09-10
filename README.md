## Telco Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to churn, enabling proactive retention strategies.

Live Demo:[https://telco-customer-churn-gdw5dyssknn3p2s7fy5fkn.streamlit.app/]

---

## Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Acquiring a new customer costs 5–7× more than retaining an existing one. This project builds a predictive model to identify at-risk customers so the business can intervene early.

Business Question: Given a customer's account details, services, and billing information, how likely are they to churn?

---

## Dataset

- Source:[IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Size: 7,043 customers × 21 features
- Target:`Churn` (Yes/No) — imbalanced: 26.5% churn rate

Features include:
- Demographics (gender, senior citizen, partner, dependents)
- Services (phone, internet, streaming, tech support)
- Account info (tenure, contract type, payment method)
- Billing (monthly charges, total charges)

---

##  Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | Scikit-learn, XGBoost |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

##  Project Workflow

### 1. Data Cleaning
- Detected hidden missing values in `TotalCharges` (stored as strings with blank spaces)
- Identified these as tenure=0 customers → filled with 0 (not mean/median)
- Removed `customerID` (unique identifier, no predictive value)

### 2. Exploratory Data Analysis (EDA)
Key insights:
- Month-to-month contracts churn at 43% vs 3% for two-year contracts
- Churned customers have lower tenure and higher monthly charges
- Fiber optic internet users churn more than DSL users

### 3. Feature Engineering
- One-hot encoded all categorical features (dropped first to avoid multicollinearity)
- Standard-scaled numeric features
- Stratified train-test split (80/20) to preserve class balance

### 4. Modeling
Trained and compared three models:

| Model | ROC-AUC | F1-Score | Recall |
|-------|---------|----------|--------|
| Logistic Regression | 0.84 | 0.61 | 0.55 |
| Random Forest | 0.82 | 0.55 | 0.48 |
| **XGBoost (tuned)** | **0.85** | **0.62** | **0.56** |

Chosen model: XGBoost, tuned with GridSearchCV

### 5. Deployment
Interactive Streamlit app where users can enter customer details and get real-time churn probability.

---

## 📁 Project Structure

telco-churn-project/
│
├── data/
│ ├── raw/ # Original Kaggle data (gitignored)
│ └── processed/ # Cleaned data (gitignored)
│
├── notebooks/
│ ├── 01_data_loading.ipynb # Load & inspect
│ ├── 02_cleaning.ipynb # Handle missing values
│ ├── 03_eda.ipynb # Visual exploration
│ └── 04_modeling.ipynb # Train, evaluate, tune
│
├── app/
│ └── streamlit_app.py # Web app
│
├── models/ # Saved model artifacts (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md


---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/hackerxanshuman/telco-customer-churn.git
cd telco-customer-churn

# 2. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate


# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Get it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place it at: data/raw/telco_churn.csv

# 5. Run the notebooks to clean data and train the model
jupyter notebook

# 6. Launch the app
cd app
streamlit run streamlit_app.py\

## Key Learnings

Handling hidden data quality issues: TotalCharges looked numeric but was stored as strings — caught via df.info().

Domain-aware imputation: Filled missing TotalCharges with 0 (not mean) because all missing rows had tenure=0.

Class imbalance: Used ROC-AUC and F1 instead of accuracy as primary metrics.

Production-minded: Separated training code from inference code, saved artifacts with joblib.

👤 Author
Anshuman
GitHub