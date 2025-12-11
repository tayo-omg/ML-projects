# MODEL 2: CLASSIFICATION - Predict High/Low Conversion


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# Load data
df = pd.read_csv("marketing_campaign.csv")

# Clean Acquisition_Cost
df["Acquisition_Cost"] = df["Acquisition_Cost"].replace(r'[\$,]', '', regex=True).astype(float)

# Feature Engineering
df["CTR"] = (df["Clicks"] / df["Impressions"]) * 100
df["CPC"] = df["Acquisition_Cost"] / df["Clicks"]

# Create binary target
df["HighConversion"] = (df["Conversion_Rate"] >= 0.08).astype(int)

# Define X and y
X = df.drop(["HighConversion", "Conversion_Rate", "Date"], axis=1)
y = df["HighConversion"]

# Categorical / Numerical
categorical_cols = X.select_dtypes(include="object").columns

# Preprocessing
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Pipeline
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", GradientBoostingClassifier())
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))





print("Predictions:", y_pred[:10])
print("Actual:", y_test[:10].values)
