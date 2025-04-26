import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

# ✅ Load the updated augmented dataset
df = pd.read_csv("data/cat_augmented.csv")

# 🔍 Extract features and labels
X = df.drop(columns=["Illness", "Test Case ID"])
y = df["Illness"]

# 🔁 Split the dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🧮 Compute sample weights to balance rare/common illness classes
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# ✅ Final AdaBoost Classifier with tuned hyperparameters
final_ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=4, min_samples_split=4, min_samples_leaf=1
    ),
    n_estimators=150,
    learning_rate=0.3,
    random_state=42,
)

# ✅ Train AdaBoost with class weights
final_ada.fit(X_train, y_train, sample_weight=sample_weights)

# 🧪 Evaluate predictions
y_pred = final_ada.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("\n✅ Final AdaBoost Model Evaluation")
print(f"📊 Accuracy: {accuracy:.4f}")
print("🧾 Classification Report:")
print(report)

# 💾 Save model and feature names
joblib.dump(final_ada, "new_cat_adaboost_model.pkl")
joblib.dump(X.columns.tolist(), "new_cat_adaboost_selected_features.pkl")

print("\n✅ Model and feature list saved successfully.")

# 🔝 Display top 20 most important features
importances = final_ada.feature_importances_
top_features = (
    pd.Series(importances, index=X.columns).sort_values(ascending=False).head(20)
)

print("\n🔝 Top 20 Most Important Features:")
for feature, score in top_features.items():
    print(f"{feature:40s} →  {score:.5f}")

# 📊 Plot feature importance
plt.figure(figsize=(12, 6))
top_features.plot(kind="barh", color="skyblue", edgecolor="black")
plt.gca().invert_yaxis()
plt.title("Top 20 Most Important Features - AdaBoost (With Class Weights)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

probs = final_ada.predict_proba(X_test)
max_probs = probs.max(axis=1)
print("🔎 Distribution of AdaBoost predicted probabilities (max per row):")
print(pd.Series(max_probs).describe())
