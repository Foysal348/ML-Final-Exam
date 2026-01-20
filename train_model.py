import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

num_cols = X.columns

# Preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols)
])

# Model
model = RandomForestClassifier(random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Train
pipeline.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("model.pkl saved successfully")
