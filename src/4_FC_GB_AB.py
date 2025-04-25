import joblib
import numpy as np
import json
import pandas as pd

# Load Knowledge Base (Rules)
def load_knowledge_base():
    with open("data/new_knowledge_base.json", "r") as kb_file:
        return json.load(kb_file)["rules"]

# Load Pretrained Models
boosting_model = joblib.load("boosting_model.pkl")
adaboost_model = joblib.load("adaboost_model.pkl")

# Load selected features used during Gradient Boosting training
selected_features = joblib.load("selected_features.pkl")

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

# Ensure all expected features exist in the dataset (missing ones should be filled with 0)
def align_features(feature_vector, expected_features):
    feature_vector = feature_vector.copy()  # Prevents SettingWithCopyWarning

    for feature in expected_features:
        if feature not in feature_vector.columns:
            feature_vector.loc[:, feature] = 0  # Correct way to add missing features

    # Reorder columns to match training feature order
    feature_vector = feature_vector[expected_features]
    return feature_vector

# Step 1: Forward Chaining - Initial Illness Matching
def forward_chaining(fact_base):
    user_symptoms = fact_base["symptoms"]

    possible_diagnoses = []

    for rule in load_knowledge_base():
        matched_symptoms = []
        total_weight = sum(symptom["weight"] for symptom in rule["symptoms"])
        matched_weight = sum(symptom["weight"] for symptom in rule["symptoms"] if symptom["name"].lower() in user_symptoms)

        if matched_weight > 0:
            match_ratio = matched_weight / total_weight
            initial_confidence = rule["confidence"] * match_ratio

            matched_symptoms = [s["name"] for s in rule["symptoms"] if s["name"].lower() in user_symptoms]

            possible_diagnoses.append({
                "illness": rule["illness"],
                "matched_symptoms": matched_symptoms,
                "confidence_fc": round(initial_confidence, 2),
            })

    return possible_diagnoses

# Step 2: Gradient Boosting - Illness Ranking Optimization
def gradient_boosting_ranking(possible_diagnoses, user_symptoms):
    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        feature_vector = pd.DataFrame(
            [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms
        )

        feature_vector_selected = feature_vector[selected_features]
        feature_vector_selected = align_features(feature_vector_selected, boosting_model.feature_names_in_)

        raw_predictions = boosting_model.predict_proba(feature_vector_selected)

        illness_index = list(boosting_model.classes_).index(diagnosis["illness"])
        gb_confidence = raw_predictions[0][illness_index]

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": round(gb_confidence, 2),
        })

    return refined_diagnoses

# Step 3: Final Confidence Adjustment with AdaBoost
def adaboost_ranking(possible_diagnoses, user_symptoms):
    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        feature_vector = pd.DataFrame(
            [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms
        )

        feature_vector_selected = feature_vector[selected_features]
        feature_vector_selected = align_features(feature_vector_selected, boosting_model.feature_names_in_)

        gb_predictions = boosting_model.predict_proba(feature_vector_selected)

        gb_predictions_df = pd.DataFrame(gb_predictions, columns=[f"GB_{cls}" for cls in boosting_model.classes_])

        feature_vector_adaboost = pd.concat([feature_vector_selected, gb_predictions_df], axis=1)

        feature_vector_adaboost = align_features(feature_vector_adaboost, adaboost_model.feature_names_in_)

        raw_predictions = adaboost_model.predict_proba(feature_vector_adaboost)
        illness_index = list(adaboost_model.classes_).index(diagnosis["illness"])

        symptom_match_ratio = len(diagnosis.get("matched_symptoms", [])) / max(len(user_symptoms), 1)

        ab_confidence = raw_predictions[0][illness_index] * (1 + symptom_match_ratio)

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis.get("matched_symptoms", []),
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": diagnosis["confidence_gb"],
            "confidence_ab": round(ab_confidence, 2),
        })

    return refined_diagnoses

# Main Execution Flow
def run_diagnosis():
    fact_base = get_user_input()

    possible_diagnoses = forward_chaining(fact_base)

    if not possible_diagnoses:
        print("\n❌ No matching illnesses found. Try entering different symptoms.")
        return

    refined_diagnoses = gradient_boosting_ranking(possible_diagnoses, fact_base["symptoms"])

    final_diagnoses = adaboost_ranking(refined_diagnoses, fact_base["symptoms"])

    final_diagnoses.sort(key=lambda x: -x["confidence_ab"])

    print("\n🩺 **Final Diagnoses (Forward Chaining + Boosting + AdaBoost):**")
    for diagnosis in final_diagnoses:
        print(f"🔹 {diagnosis['illness']} (Confidence: {diagnosis['confidence_ab']})")

# Run the system
if __name__ == "__main__":
    run_diagnosis()
