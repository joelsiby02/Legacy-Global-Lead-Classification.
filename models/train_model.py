import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("../data/leads.csv")

print("Dataset loaded:", df.shape)
print(df.head())


# -----------------------------
# 2. Define Features and Target
# -----------------------------
X = df.drop(columns=["deposit", "lead_id"])
y = df["deposit"]


# -----------------------------
# 3. Define Feature Types
# -----------------------------
categorical_features = [
    "state",
    "profession",
    "lead_source"
]

numeric_features = [
    "city_tier",
    "age",
    "answered_call",
    "asked_about_leverage",
    "demo_requested"
]


# -----------------------------
# 4. Preprocessing Pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# -----------------------------
# 5. Model
# -----------------------------
model = LogisticRegression(max_iter=1000)


# -----------------------------
# 6. Full Pipeline
# -----------------------------
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])


# -----------------------------
# 7. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 8. Train Model
# -----------------------------
pipeline.fit(X_train, y_train)

print("\nModel training completed.")


# -----------------------------
# 9. Evaluate Model
# -----------------------------
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Evaluation")
print("--------------------")
print("Accuracy:", accuracy)
print("\nClassification Report")
print(classification_report(y_test, predictions))


# -----------------------------
# 10. Feature Importance (for demo)
# -----------------------------
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

coefficients = pipeline.named_steps["model"].coef_[0]

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": coefficients
})

importance_df = importance_df.sort_values(by="importance", ascending=False)

print("\nTop Lead Signals Learned by Model:")
print("----------------------------------")
print(importance_df.head(10))


# -----------------------------
# 11. Save Model
# -----------------------------
joblib.dump(pipeline, "lead_scoring_model.pkl")

print("\nModel saved as: lead_scoring_model.pkl")