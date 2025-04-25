import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

# 📌 Load dataset and knowledge base
df = pd.read_csv("latest_augmented.csv")
with open("data/updated_knowledge_base_v2_fixed.json", "r") as f:
    kb = json.load(f)["rules"]

# 📌 Build illness-to-category mapping from KB
illness_to_type = {entry["illness"]: entry.get("type", "Unknown") for entry in kb}
df["Category"] = df["Illness"].map(illness_to_type)

# 🧼 Remove rows with missing category info
df = df.dropna(subset=["Category"])

# 🔍 Extract features and new labels
X = df.drop(columns=["Illness", "Category"])
y = df["Category"]

# 🔁 Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🧮 Compute balanced sample weights
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# ✅ Configure and train AdaBoost classifier (for categories)
ada_category_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=4, min_samples_split=4, min_samples_leaf=1
    ),
    n_estimators=150,
    learning_rate=0.3,
    random_state=42,
)

ada_category_model.fit(X_train, y_train, sample_weight=sample_weights)

# 🧪 Evaluate results
y_pred = ada_category_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 💾 Save model and selected features
joblib.dump(ada_category_model, "adaboost_category_model.pkl")
joblib.dump(X.columns.tolist(), "adaboost_category_selected_features.pkl")

# 📁 Save mapping from category → illnesses
category_to_illnesses = {}
for illness, illness_type in illness_to_type.items():
    category_to_illnesses.setdefault(illness_type, []).append(illness)

with open("category_to_illnesses.json", "w") as f:
    json.dump(category_to_illnesses, f, indent=2)

# 🖨️ Show evaluation
print("\n✅ AdaBoost Category Model Evaluation")
print(f"📊 Accuracy: {accuracy:.4f}")
print("🧾 Classification Report:")
print(report)
print(df["Category"].value_counts())

# 📈 Feature importance
importances = ada_category_model.feature_importances_
top_features = (
    pd.Series(importances, index=X.columns).sort_values(ascending=False).head(20)
)

print("\n🔝 Top 20 Most Important Features:")
for feature, score in top_features.items():
    print(f"{feature:40s} →  {score:.5f}")

# 📊 Plot
plt.figure(figsize=(10, 6))
top_features.plot(kind="barh", color="teal", edgecolor="black")
plt.gca().invert_yaxis()
plt.title("Top 20 Most Important Features - AdaBoost (Category Classification)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("top_20_features_category_adaboost.png")
plt.show()
