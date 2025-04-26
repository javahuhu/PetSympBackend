import pandas as pd
import joblib

# Load the dataset and the model
df = pd.read_csv("data/cat_augmented.csv")
adaboost_model = joblib.load("new model/new_cat_adaboost_model.pkl")
selected_features = joblib.load("new model/new_cat_adaboost_selected_features.pkl")

# Prepare the input features
X = df[selected_features]

# Make predictions
df["Predicted"] = adaboost_model.predict(X)

# Save true vs predicted labels
df[["Illness", "Predicted"]].to_csv("cat_true_vs_predicted.csv", index=False)

print("✅ Prediction file saved as cat_true_vs_predicted.csv")
