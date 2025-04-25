import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the extracted dataset
df = pd.read_csv("data/extracted_dataset.csv")

# Encode the target variable (Illness) as numerical values
label_encoder = LabelEncoder()
df["Illness"] = label_encoder.fit_transform(df["Illness"])

# Separate features and target
X = df.drop(columns=["Illness"])  # Feature variables
y = df["Illness"]  # Target variable

# Check illness distribution
illness_counts = y.value_counts()

# If any illness appears only once, disable stratify and ensure all rare illnesses go to training set
if illness_counts.min() < 2:
    print(
        "⚠ Warning: Some illnesses have only 1 occurrence. Assigning all rare illnesses to training set."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


# Train Gradient Boosting Model with improved settings
gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, subsample=0.8, random_state=42
)
gb_model.fit(X_train, y_train)

# Train AdaBoost Model with improved settings
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ada_model.fit(X_train, y_train)

# Make predictions
gb_predictions = gb_model.predict(X_test)
ada_predictions = ada_model.predict(X_test)

# Evaluate the models
gb_accuracy = accuracy_score(y_test, gb_predictions)
ada_accuracy = accuracy_score(y_test, ada_predictions)

# Print model performance
print("Gradient Boosting Accuracy:", gb_accuracy)
print("AdaBoost Accuracy:", ada_accuracy)

# Fix classification report issue by ensuring correct label mapping and preventing errors
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
