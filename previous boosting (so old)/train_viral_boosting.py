import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE  # For handling rare illnesses
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("updated_viral_diseases_dataset.csv")

# Encode the target variable (Illness) into numerical values
label_encoder = LabelEncoder()
df["Illness"] = label_encoder.fit_transform(df["Illness"])

# Separate features and target
X = df.drop(columns=["Illness"])  # Features (symptoms, severity, etc.)
y = df["Illness"]  # Target variable (illness classification)

# Convert Severity to numeric values before scaling
severity_mapping = {"Low": 1, "Medium": 2, "High": 3}
X["Severity"] = X["Severity"].map(severity_mapping)

# Convert Stage columns (e.g., Vomiting_Stage) to numeric values
for col in X.columns:
    if "_Stage" in col:
        X[col] = X[col].map(
            {"Early": 0, "Any Stage": 1, "Late": 2, "None": -1}
        )  # Encoding stages

# Identify missing values before training
print("Missing Values per Column:\n", X.isnull().sum())

# Handle NaN values by filling them with 0
X.fillna(0, inplace=True)  # Replace ALL NaNs with 0

# Normalize numerical features (Severity, Priority, Confidence)
scaler = StandardScaler()
X[["Severity", "Priority", "Confidence"]] = scaler.fit_transform(
    X[["Severity", "Priority", "Confidence"]]
)

# Apply SMOTE if any illness is underrepresented
illness_counts = y.value_counts()
if illness_counts.min() < 5:
    print("⚠ Warning: Some illnesses are rare. Applying SMOTE to balance classes.")
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X, y = smote.fit_resample(X, y)

# Split dataset into Training (80%) & Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Gradient Boosting Model
gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, subsample=0.8, random_state=42
)
gb_model.fit(X_train, y_train)

# Train AdaBoost Model
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ada_model.fit(X_train, y_train)

# Make predictions
gb_predictions = gb_model.predict(X_test)
ada_predictions = ada_model.predict(X_test)

# Evaluate the models
gb_accuracy = accuracy_score(y_test, gb_predictions)
ada_accuracy = accuracy_score(y_test, ada_predictions)

# Print model performance
print("\nGradient Boosting Accuracy:", gb_accuracy)
print("AdaBoost Accuracy:", ada_accuracy)

# Fix classification report issue by ensuring correct label mapping
print("\nGradient Boosting Classification Report:")
print(
    classification_report(
        y_test,
        gb_predictions,
        labels=np.unique(y_test),
        target_names=label_encoder.inverse_transform(np.unique(y_test)),
        zero_division=0,
    )
)

print("\nAdaBoost Classification Report:")
print(
    classification_report(
        y_test,
        ada_predictions,
        labels=np.unique(y_test),
        target_names=label_encoder.inverse_transform(np.unique(y_test)),
        zero_division=0,
    )
)

import joblib

# Save the label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# Save the trained models
joblib.dump(gb_model, "gradient_boosting_model.pkl")
joblib.dump(ada_model, "adaboost_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")  # Save feature names

print("✅ Models and label encoder saved successfully!")
