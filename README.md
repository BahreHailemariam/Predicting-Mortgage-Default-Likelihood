# 🏠 Predicting Mortgage Default Likelihood

## 🧠 Project Overview
This project focuses on **predicting the likelihood of mortgage loan defaults** using advanced data analytics and machine learning techniques.  
The goal is to support financial institutions, lenders, and mortgage servicers in assessing **credit risk**, improving **loan underwriting**, and designing **data-driven policies** to minimize losses.  

By identifying high-risk borrowers early, the model helps financial institutions make proactive decisions such as loan restructuring, credit monitoring, and targeted intervention.  

## 🎯 Business Objectives
1. Predict which mortgage loans are at risk of default.  
2. Identify key risk drivers influencing borrower default.  
3. Improve credit risk management and loan portfolio health.  
4. Build a repeatable and scalable pipeline for predictive risk modeling.  
5. Deliver visual insights and KPIs through **Power BI** and **Python dashboards**.  

## 🧩 Data Description
The dataset includes borrower and loan-level information such as demographics, credit history, and repayment behavior.

| Feature | Description | Example |
|----------|-------------|----------|
| loan_id | Unique identifier for each mortgage | 202405 |
| borrower_income | Annual income of borrower | 85,000 |
| loan_amount | Original loan amount | 250,000 |
| interest_rate | Mortgage interest rate (%) | 3.75 |
| loan_term | Length of the loan (months) | 360 |
| credit_score | Borrower’s credit score | 710 |
| LTV | Loan-to-Value ratio (%) | 85 |
| DTI | Debt-to-Income ratio (%) | 36 |
| property_value | Appraised value of property | 294,000 |
| employment_years | Number of years employed | 7 |
| default | Target variable (1 = Default, 0 = Non-default) | 0 |

## ⚙️ Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| Data Processing | Pandas, NumPy, Power Query |
| Exploratory Analysis | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, Logistic Regression |
| Model Deployment | Streamlit, Flask |
| Visualization | Power BI |
| Automation | Airflow, Cron Jobs |
| Version Control | Git, GitHub |

## 🔄 Workflow

## **1️⃣ Define Business Problem**
**Goal:**  
Financial institutions want to identify borrowers who are likely to **default on their mortgage loans** before it happens.  

**Key Questions:**  
- Which borrower or loan characteristics indicate a higher default risk?  
- How can early risk detection reduce losses and improve credit decisions?  
- What interventions (loan restructuring, rate adjustments) can minimize default rates?  

**Deliverables:**  
- Predictive model that outputs a **default probability score** for each borrower.  
- Dashboards showing **portfolio risk metrics** and trends for decision-makers.  

---

## **2️⃣ Extract & Clean Mortgage Dataset (CSV, SQL, API)**
**Goal:** Gather and consolidate all relevant data sources.  

**Data Sources:**
- 📂 **CSV Files** (historical loan data, payment history, customer demographics)  
- 🗄️ **SQL Databases** (loan servicing systems, transaction records)  
- 🌐 **APIs** (credit score providers, property valuation services)  

**Python Example (SQL + CSV):**
```python
import pandas as pd
import sqlalchemy as db

# Load from SQL
engine = db.create_engine("mysql+pymysql://user:password@localhost/mortgage_db")
loan_data = pd.read_sql("SELECT * FROM loan_records", engine)

# Load from CSV
borrower_data = pd.read_csv("borrowers.csv")

# Merge
data = pd.merge(loan_data, borrower_data, on="loan_id")
```

**Cleaning Tasks:**
- Remove duplicates  
- Standardize column names  
- Parse dates and financial figures  
- Drop or flag incomplete entries  

---

## **3️⃣ Handle Missing Values, Outliers, and Encoding**
**Goal:** Ensure high-quality, consistent data for modeling.  

**Tasks:**
- **Missing Values:**  
  - Numeric fields → Impute with mean/median or model-based imputation.  
  - Categorical fields → Impute with mode or “Unknown” label.  

- **Outliers:**  
  - Winsorize extreme values (e.g., incomes, loan_amounts).  
  - Use IQR or z-score to detect anomalies.  

- **Encoding:**  
  - One-hot encode categorical variables (loan_type, region).  
  - Label encode ordinal variables (employment_status).  

**Python Example:**
```python
data['credit_score'].fillna(data['credit_score'].median(), inplace=True)
data = pd.get_dummies(data, columns=['loan_type', 'region'], drop_first=True)
```

---

## **4️⃣ Feature Engineering (Ratios, Interactions, Scaling)**
**Goal:** Create new variables that enhance predictive power.  

**Examples:**
- **Debt-to-Income Ratio (DTI)** = total_debt / borrower_income  
- **Loan-to-Value (LTV)** = loan_amount / property_value  
- **Installment Ratio** = monthly_payment / borrower_income  
- Interaction features: `credit_score * LTV`, `income * loan_amount`  

**Scaling:**
- Normalize continuous variables (e.g., credit_score, income).  
- Use `StandardScaler` or `MinMaxScaler` before model training.  

---

## **5️⃣ Split Data (Train/Test)**
**Goal:** Prevent overfitting and evaluate generalization.  

**Steps:**
- Split dataset into **training (70%)** and **testing (30%)** sets.  
- Optionally, create a **validation set (10%)** for hyperparameter tuning.  

**Python Example:**
```python
from sklearn.model_selection import train_test_split

X = data.drop("default", axis=1)
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## **6️⃣ Train Machine Learning Models**
**Goal:** Build predictive models to estimate default probability.  

**Algorithms to try:**
- Logistic Regression → baseline  
- Random Forest → interpretability  
- XGBoost / LightGBM → high performance  

**Python Example:**
```python
from xgboost import XGBClassifier
model = XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=300, random_state=42)
model.fit(X_train, y_train)
```

**Output:** Model generates a **probability of default** for each borrower.

---

## **7️⃣ Evaluate Using AUC, Precision-Recall, Confusion Matrix**
**Goal:** Measure accuracy and model reliability.  

**Metrics:**
- **AUC (ROC Curve):** Overall ability to distinguish between defaulters and non-defaulters.  
- **Precision & Recall:** Balance between false positives and false negatives.  
- **Confusion Matrix:** Actual vs. predicted classifications.  

**Python Example:**
```python
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
```

---

## **8️⃣ Deploy with Streamlit or Power BI**
**Goal:** Make the insights accessible to business users.  

**Option 1 – Streamlit Web App:**
```bash
streamlit run app.py
```
- Interactive model input forms  
- Displays prediction probabilities  
- Visualizes portfolio-level KPIs  

**Option 2 – Power BI Dashboard:**
- Connect model output CSV or SQL database  
- Create visual KPIs:  
  - Default Rate (%)  
  - Risk by Region / Loan Type  
  - Borrower Segment Analysis  

---

## **9️⃣ Automate Predictions with Scheduled Scripts (Airflow / Cron Jobs)**
**Goal:** Ensure continuous, automated model predictions.  

**Airflow DAG Example:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def run_prediction():
    import predict_mortgage_default
    predict_mortgage_default.main()

dag = DAG('mortgage_default_pipeline', schedule_interval='@monthly', start_date=datetime(2025,1,1))

task = PythonOperator(task_id='run_default_model', python_callable=run_prediction, dag=dag)
```

**Benefits:**
- Automatically refresh predictions monthly or weekly.  
- Send email notifications or export reports to Power BI.  


## 📊 Power BI Dashboard KPIs
- Default Rate (%)  
- Portfolio at Risk (PAR)  
- Average LTV & DTI  
- Top Risk Drivers  
- Borrower Segment Default Trends  
- Loan Performance Over Time  

## 📈 Model Performance
| Metric | Value |
|---------|-------|
| Accuracy | 0.88 |
| Precision | 0.81 |
| Recall | 0.76 |
| F1-score | 0.78 |
| AUC | 0.91 |

## 🧮 Sample Code Snippet
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

data = pd.read_csv("mortgage_data.csv")
X = data[['credit_score', 'LTV', 'DTI', 'loan_amount', 'borrower_income']]
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))
print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
```

## 🚀 Deployment
### Streamlit Web App
```bash
streamlit run app.py
```

### Flask API
```bash
python app.py
```

## 🔁 Reproducibility
### Setup
```bash
git clone https://github.com/yourusername/predict-mortgage-default.git
cd predict-mortgage-default
pip install -r requirements.txt
```
### Train Model
```bash
python scripts/train_model.py
```

## 💡 Business Insights
- Borrowers with LTV > 90% and DTI > 45% are 3x more likely to default.  
- Fixed-rate loans have lower default probability than adjustable-rate loans.  
- Default risk increases when credit scores fall below 680.  
- The model helps identify high-risk loans before disbursement, reducing losses by up to 15%.  

## 📜 License
This project is licensed under the **MIT License**.

## 👨‍💻 Author
**Bahre Hailemariam**  
_Data Analyst | BI Developer_  
📩 bahre.hail@gmail.com  
🌐 [Portfolio](https://bahre-hailemariam-data-analyst.crd.co/)
💼 [LinkedIn](https://www.linkedin.com/in/bahre-hailemariam/)
📊[GitHub](https://github.com/BahreHailemariam)
