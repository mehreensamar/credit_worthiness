import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/cs-training.csv", index_col=0)

# Preprocessing
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(0, inplace=True)
df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 1]

# Features & Labels
X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

# SMOTE
X_res, y_res = SMOTE().fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
