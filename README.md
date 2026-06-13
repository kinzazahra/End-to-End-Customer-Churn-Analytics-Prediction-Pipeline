# End-to-End Customer Churn Analytics Pipeline

---

## Overview

This project builds a complete Customer Churn Analytics Pipeline using machine learning to predict whether a customer is likely to leave a telecom service. The workflow covers data acquisition, preprocessing, exploratory analysis, feature engineering, model training, evaluation, and business insights generation.

---

## Objectives

- Analyze customer behavior and churn patterns.
- Engineer meaningful features from customer data.
- Train a machine learning model to predict churn.
- Evaluate model performance using classification metrics.
- Identify the most important factors influencing customer churn.
- Generate actionable business insights for customer retention.

---

## Dataset

The project uses the **Telco Customer Churn** dataset from OpenML.

### Target Variable

- **Churn**
  - 0 → Customer Retained
  - 1 → Customer Churned

### Dataset Features

The dataset contains customer information such as:

- Customer demographics
- Service subscriptions
- Contract details
- Monthly charges
- Total charges
- Tenure
- Payment methods

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenML

---

## Project Workflow

### 1. Data Collection

The dataset is fetched directly from OpenML:

```python
from sklearn.datasets import fetch_openml

data = fetch_openml(
    name="Telco-Customer-Churn",
    version=1,
    as_frame=True
)
```

---

### 2. Data Exploration

Performed:

- Shape inspection
- Data type analysis
- Statistical summaries
- Missing value detection

```python
print(df.shape)
print(df.info())
print(df.describe())
```

---

### 3. Data Cleaning

Missing values are handled using forward-fill imputation.

```python
df = df.fillna(method="ffill")
```

---

### 4. Data Encoding

Categorical variables are converted into numerical format using Label Encoding.

```python
from sklearn.preprocessing import LabelEncoder
```

---

### 5. Exploratory Data Analysis

A correlation heatmap is generated to understand relationships among variables.

```python
sns.heatmap(df.corr())
```

---

### 6. Feature Engineering

Additional business-focused features are created:

#### Monthly Charge Per Tenure

```python
MonthlyCharges / (tenure + 1)
```

#### Total Charge Per Service Duration

```python
TotalCharges / (tenure + 1)
```

These features help capture spending behavior relative to customer lifespan.

---

### 7. Train-Test Split

Dataset is divided into:

- 80% Training Data
- 20% Testing Data

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 8. Model Training

A Random Forest Classifier is used for churn prediction.

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

```python
classification_report()
confusion_matrix()
```

---

## Visualizations

### Correlation Matrix

Helps identify relationships among features.

### Confusion Matrix

Shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

### Feature Importance

Displays the top factors contributing to churn prediction.

---

## Business Insights Generated

The pipeline identifies:

- High-risk customers
- Important churn drivers
- Customer retention opportunities
- Revenue-risk indicators

---

## Expected Output

### Model Accuracy

```text
Accuracy: XX.XX%
```

### Classification Report

```text
Precision
Recall
F1-Score
Support
```

### Top Churn Drivers

```text
Tenure
Monthly Charges
Contract Type
Total Charges
Service Usage Features
```

---

## Project Structure

```text
Customer-Churn-Analytics/
│
├── end_to_end_customer_churn_analytics_pipeline.ipynb
├── README.md
└── requirements.txt
```

---

## Key Features

- Automated data acquisition
- Data cleaning and preprocessing
- Feature engineering
- Churn prediction using Random Forest
- Business-focused analytics
- Feature importance analysis
- Customer retention insights
- End-to-end machine learning workflow

---

## Future Enhancements

- XGBoost and LightGBM models
- Hyperparameter optimization
- SHAP explainability analysis
- Real-time churn prediction API
- Streamlit dashboard
- Customer segmentation analytics
- Automated reporting system

---

## Use Cases

- Telecom customer retention
- Subscription business analytics
- Customer success monitoring
- Revenue protection strategies
- Predictive business intelligence

---

## Author

**Kinza Zahra**
