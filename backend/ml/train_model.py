# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ================================
# 2. Load Dataset
# ================================

df = pd.read_csv("/Users/stutisamanta/Desktop/financial_risk_management/data/credit_risk_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ================================
# 3. Check Missing Values
# ================================

print(df.isnull().sum())

# Fill missing values
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

# ================================
# 4. Encode Categorical Features
# ================================

categorical_cols = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ================================
# 5. Define Features and Target
# ================================

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# ================================
# 6. Train Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ================================
# 7. Train Models
# ================================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ================================
# 8. Predictions
# ================================

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# ================================
# 9. Evaluation
# ================================

print("----- Logistic Regression -----")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\n----- Random Forest -----")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
import joblib
joblib.dump(rf, "backend/ml/models/credit_risk_model.pkl")