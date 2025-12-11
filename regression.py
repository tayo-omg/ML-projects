# MODEL 1: REGRESSION - Predict ROI

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Data
df = pd.read_csv("marketing_campaign.csv")

# Clean Acquisition_Cost
df["Acquisition_Cost"] = df["Acquisition_Cost"].replace(r'[\$,]', '', regex=True).astype(float)


df["CTR"] = (df["Clicks"] / df["Impressions"]) * 100
df["CPC"] = df["Acquisition_Cost"] / df["Clicks"]

# Define features and target
X = df.drop(["ROI", "Date"], axis=1)
y = df["ROI"]

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

# Preprocessing
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Model pipeline
model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))





print("Sample Predictions:")
print(y_pred[:10])
