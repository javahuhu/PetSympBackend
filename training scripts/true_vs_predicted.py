import pandas as pd
import joblib

# Load the dataset and the model
df = pd.read_csv("latest_augmented.csv")
adaboost_model = joblib.load("adaboost_model.pkl")
selected_features = joblib.load("adaboost_selected_features.pkl")

# Prepare the input features
X = df[selected_features]

# Make predictions
df["Predicted"] = adaboost_model.predict(X)

# Save true vs predicted labels
df[["Illness", "Predicted"]].to_csv("true_vs_predicted.csv", index=False)

print("✅ Prediction file saved as true_vs_predicted.csv")
