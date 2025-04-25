import joblib
import numpy as np
import pandas as pd

# Load the trained models
gb_model = joblib.load("gradient_boosting_model.pkl")
ada_model = joblib.load("adaboost_model.pkl")
feature_names = joblib.load("feature_names.pkl")  # Load saved feature order

print("✅ Models and feature names loaded successfully!")

# Define symptom list (same as training)
all_symptoms = [
    "Abdominal Pain",
    "Aggression",
    "Bleeding Disorders",
    "Bloody Stool",
    "Chest Pain",
    "Congested Lungs",
    "Coughing",
    "Dehydration",
    "Depression",
    "Diarrhea",
    "Difficulty Breathing",
    "Difficulty Swallowing",
    "Drooling",
    "Eye Discharge",
    "Fever",
    "Forceful Cough",
    "Gagging",
    "Hydrophobia",
    "Itching",
    "Jaundice",
    "Lack of Coordination",
    "Lethargy",
    "Loss of Appetite",
    "Muscle Spasms",
    "Nasal Discharge",
    "Paralysis",
    "Pale Gums",
    "Runny Nose",
    "Seizures",
    "Sensitivity to Light",
    "Sneezing",
    "Tender Abdomen",
    "Thick Substance in Eyes",
    "Vomiting",
    "Weakness",
    "Weight Loss",
]


def prepare_input_vector(symptom_input):
    """
    Converts user symptom input into a correctly formatted feature vector matching the model.
    """
    # Create a DataFrame with the correct feature structure
    input_vector = pd.DataFrame(columns=feature_names)  # Match training feature order

    # Initialize all features to 0
    input_vector.loc[0] = 0

    # Set symptoms to 1 if they are present
    for symptom in symptom_input:
        if symptom in input_vector.columns:
            input_vector.at[0, symptom] = 1

    # Set missing numerical features (Severity, Priority, Confidence) to neutral values
    if "Severity" in input_vector.columns:
        input_vector["Severity"] = 2  # Neutral (Medium)
    if "Priority" in input_vector.columns:
        input_vector["Priority"] = 1.5  # Average priority
    if "Confidence" in input_vector.columns:
        input_vector["Confidence"] = 0.85  # Default confidence level

    # Set missing stage features (_Stage) to "Any Stage"
    for col in input_vector.columns:
        if "_Stage" in col:
            input_vector[col] = 1  # Assume Any Stage as default

    return input_vector


def predict_illness(symptom_input, model):
    """
    Function to classify illness based on user symptoms using a trained ML model.
    """
    symptom_vector = prepare_input_vector(
        symptom_input
    )  # Ensure correct feature structure
    predicted_illness_index = model.predict(symptom_vector)[0]

    # Load label encoder to convert index back to illness name
    label_encoder = joblib.load("label_encoder.pkl")  # Load saved label encoder
    predicted_illness = label_encoder.inverse_transform([predicted_illness_index])[0]

    return predicted_illness


# Example Usage
user_symptoms = ["Coughing", "Nasal Discharge", "Sneezing"]
predicted_illness = predict_illness(user_symptoms, gb_model)
print("Predicted Illness:", predicted_illness)
