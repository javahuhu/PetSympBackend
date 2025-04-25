import joblib
import numpy as np
import json
import pandas as pd

# Load Knowledge Base (Rules)
def load_knowledge_base():
    with open("data/new_knowledge_base.json", "r") as kb_file:
        return json.load(kb_file)["rules"]

# Load Pretrained PKL Boosting Model
boosting_model = joblib.load("boosting_model.pkl")

# Load symptom list (to ensure feature vector consistency)
df = pd.read_csv("canine_illness_dataset.csv")
all_symptoms = [col for col in df.columns if col not in ["Test Case ID", "Illness"]]

# Get User Input
def get_user_input():
    print("\n🔹 Enter your pet's symptoms (comma-separated, e.g., Vomiting, Loss of Appetite):")
    user_symptoms = input("➡ Symptoms: ").strip().split(",")
    user_symptoms = [symptom.strip().lower() for symptom in user_symptoms]

    # Get pet details
    age_range = input("➡ Pet Age Range (Puppy, Adult, Senior): ").strip().capitalize()
    breed = input("➡ Pet Breed (or type 'Any'): ").strip().capitalize()
    size = input("➡ Pet Size (Small, Medium, Large): ").strip().capitalize()

    return {
        "pet_info": {"age_range": age_range, "breed": breed, "size": size},
        "symptoms": user_symptoms,
    }

# Extract feature importance from trained boosting model
feature_importance = boosting_model.feature_importances_

# Normalize importance scores to scale from 1.0 to 1.5
min_imp = np.min(feature_importance)
max_imp = np.max(feature_importance)
scaled_importance = 1.0 + (feature_importance - min_imp) / (max_imp - min_imp) * 0.5

# Compute symptom rarity (how many illnesses each symptom appears in)
knowledge_base = load_knowledge_base()
# Compute symptom rarity (how many illnesses each symptom appears in)
symptom_count = {symptom.lower().strip(): 0 for symptom in all_symptoms}  # Ensure uniformity

for rule in knowledge_base:
    for symptom in rule["symptoms"]:
        symptom_name = symptom["name"].lower().strip()  # Standardize symptom format
        if symptom_name in symptom_count:  # Only count symptoms that exist in all_symptoms
            symptom_count[symptom_name] += 1  # Count how many illnesses have this symptom

# Convert to a rarity score (lower count = higher rarity boost)
symptom_rarity = {symptom: 1.0 + (1 / (count + 1) * 0.5) for symptom, count in symptom_count.items()}


# Step 1: Forward Chaining - Initial Illness Matching
def forward_chaining(fact_base):
    user_symptoms = fact_base["symptoms"]
    pet_info = fact_base["pet_info"]

    possible_diagnoses = []

    for rule in knowledge_base:
        matched_symptoms = []
        total_weight = sum(symptom["weight"] for symptom in rule["symptoms"])
        matched_weight = sum(symptom["weight"] for symptom in rule["symptoms"] if symptom["name"].lower() in user_symptoms)

        if matched_weight > 0:
            match_ratio = matched_weight / total_weight
            initial_confidence = rule["confidence"] * match_ratio

            possible_diagnoses.append({
                "illness": rule["illness"],
                "matched_symptoms": [s["name"] for s in rule["symptoms"] if s["name"].lower() in user_symptoms],
                "confidence_fc": round(initial_confidence, 2),
            })

    # 🔹 Step 2: Apply Feature Importance and Symptom Rarity Boost
    for diagnosis in possible_diagnoses:
        for idx, symptom in enumerate(all_symptoms):
            if symptom in user_symptoms:
                rarity_boost = symptom_rarity.get(symptom, 1.0)  # Get rarity boost
                diagnosis["confidence_fc"] *= scaled_importance[idx] * rarity_boost  # Apply both adjustments

    return possible_diagnoses

# Step 2: Gradient Boosting - Illness Ranking Optimization (PKL Model)
def gradient_boosting_ranking(possible_diagnoses, user_symptoms):
    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # Create a feature vector for Boosting
        feature_vector = pd.DataFrame(
            [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms  # Ensure feature names match training data
        )

        # Predict Adjusted Confidence using Boosting Model
        raw_predictions = boosting_model.predict_proba(feature_vector)

        # Get the correct index of the illness in the model's class order
        illness_index = list(boosting_model.classes_).index(diagnosis["illness"])

        # Retrieve the probability for the correct illness
        gb_confidence = raw_predictions[0][illness_index]

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": round(gb_confidence, 2),
        })

    return refined_diagnoses

# Main Execution Flow
def run_diagnosis():
    fact_base = get_user_input()

    # Step 1: Run Forward Chaining
    possible_diagnoses = forward_chaining(fact_base)

    if not possible_diagnoses:
        print("\n❌ No matching illnesses found. Try entering different symptoms.")
        return

    # Step 2: Optimize Illness Ranking with Gradient Boosting
    refined_diagnoses = gradient_boosting_ranking(possible_diagnoses, fact_base["symptoms"])

    # Sort illnesses by highest confidence
    refined_diagnoses.sort(key=lambda x: -x["confidence_gb"])

    # Print Results
    print("\n🩺 **Final Diagnoses (Forward Chaining + Boosting):**")
    for diagnosis in refined_diagnoses:
        print(f"🔹 {diagnosis['illness']} (Confidence: {diagnosis['confidence_gb']})")

# Run the system
if __name__ == "__main__":
    run_diagnosis()
