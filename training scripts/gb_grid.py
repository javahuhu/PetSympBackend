import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# 📌 Load dataset
df = pd.read_csv("latest.csv")

# 📌 Separate features and target
X = df.drop(columns=["Illness"])
y = df["Illness"]

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 📌 Define the parameter grid for tuning
param_grid = {
    "n_estimators": [50, 75, 100],
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [3, 4, 5],
    "min_samples_split": [2, 4, 6],
    "subsample": [0.8, 0.85, 1.0],
    "max_features": ["sqrt", "log2"],
}

# 🔍 Initialize the model
gb_model = GradientBoostingClassifier(random_state=42)

# 🔍 Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    verbose=2,
    n_jobs=-1,
)

# 🔹 Fit grid search to the data
grid_search.fit(X_train, y_train)

# ✅ Output the best parameters
print("\n📌 Best Parameters Found:")
print(grid_search.best_params_)
print(f"📈 Best Cross-Validated F1 Score: {grid_search.best_score_:.4f}")

# 🔁 Train final model using best parameters
best_params = grid_search.best_params_
final_model = GradientBoostingClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# 💾 Save the trained model
joblib.dump(final_model, "gradient_model.pkl")
print("✅ Final model saved as 'gradient_model.pkl'")

# 📊 Make predictions on test set
y_pred = final_model.predict(X_test)

# 📈 Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\n📊 Evaluation Metrics on Test Set:")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall:    {recall:.4f}")
print(f"✅ F1 Score:  {f1:.4f}")
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 📌 Feature Importance
feature_importance = final_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# 📁 Save full feature importance as JSON
importance_dict = {
    X.columns[i]: float(feature_importance[i]) for i in range(len(X.columns))
}
with open("gradient_feature_importance.json", "w") as f:
    json.dump(importance_dict, f, indent=2)
print("✅ Feature importance saved as 'gradient_feature_importance.json'")

# 📊 Plot Top 20 Important Features
top_n = 20
top_features = [X.columns[i] for i in sorted_idx[:top_n]]
top_importance = feature_importance[sorted_idx[:top_n]]

plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importance[::-1])
plt.xlabel("Feature Importance Score")
plt.ylabel("Symptoms")
plt.title("Top 20 Most Important Symptoms in Illness Classification")
plt.tight_layout()
plt.show()
