from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
import pandas as pd
import os

app = Flask(__name__)

# Path to store the results
FACT_BASE_PATH = "data/fact_base.json"

# Load Knowledge Base (Rules)
def load_knowledge_base():
    with open("data/new_knowledge_base.json", "r") as kb_file:
        return json.load(kb_file)["rules"]

# Load Illness Information Database
def load_illness_info():
    with open("data/illness_info.json", "r") as illness_file:
        return json.load(illness_file)

# Load Pretrained Models
boosting_model = joblib.load("boosting_model.pkl")
adaboost_model = joblib.load("adaboost_model.pkl")

# Load selected features used during Gradient Boosting training
selected_features = joblib.load("selected_features.pkl")

# Load symptom list (to ensure feature vector consistency)
df = pd.read_csv("canine_illness_dataset.csv")
all_symptoms = [col for col in df.columns if col not in ["Test Case ID", "Illness"]]

# Categorize pet age
def categorize_age(age):
    age = int(age)  # Ensure it's an integer
    if age <= 1:
        return "Puppy"
    elif 1 < age <= 7:
        return "Adult"
    else:
        return "Senior"

# Ensure all expected features exist in the dataset (missing ones should be filled with 0)
def align_features(feature_vector, expected_features):
    for feature in expected_features:
        if feature not in feature_vector.columns:
            feature_vector.loc[:, feature] = 0
    return feature_vector[expected_features]

# Step 1: Forward Chaining - Initial Illness Matching
def forward_chaining(fact_base):
    user_symptoms = [s["name"].lower().strip() for s in fact_base["symptoms"]]
    pet_info = fact_base["pet_info"]

    possible_diagnoses = []
    for rule in load_knowledge_base():
        matched_symptoms = []
        total_weight = sum(symptom["weight"] for symptom in rule["symptoms"])
        matched_weight = sum(symptom["weight"] for symptom in rule["symptoms"] if symptom["name"].lower() in user_symptoms)

        if matched_weight > 0:
            match_ratio = matched_weight / total_weight
            initial_confidence = rule["confidence"] * match_ratio

            # Apply pet filtering
            age_range = rule.get("age_range", "Any").lower()
            breed = rule.get("breed", "Any").lower()
            size = rule.get("size", "Any").lower()

            pet_age_range = pet_info.get("age_range", "Any").lower()
            pet_breed = pet_info.get("breed", "Any").lower()
            pet_size = pet_info.get("size", "Any").lower()

            # Default multipliers
            age_match, breed_match, size_match = 1.0, 1.0, 1.0

            if age_range != "any" and age_range != pet_age_range:
                age_match = 0.85
            elif age_range == pet_age_range:
                age_match = 1.2  

            if breed != "any" and breed != pet_breed:
                breed_match = 0.9
            elif breed == pet_breed:
                breed_match = 1.1

            if size != "any" and size != pet_size:
                size_match = 0.9
            elif size == pet_size:
                size_match = 1.1

            final_confidence = round(initial_confidence * age_match * breed_match * size_match, 2)

            possible_diagnoses.append({
                "illness": rule["illness"],
                "matched_symptoms": [s["name"] for s in rule["symptoms"] if s["name"].lower() in user_symptoms],
                "confidence_fc": final_confidence,
            })

    return possible_diagnoses

# Step 2: Gradient Boosting - Illness Ranking Optimization
def gradient_boosting_ranking(possible_diagnoses, fact_base):
    user_symptoms = [s["name"].lower().strip() for s in fact_base["symptoms"]]
    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        feature_vector = pd.DataFrame(
            [[1 if symptom in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms
        )
        feature_vector_selected = align_features(feature_vector[selected_features], boosting_model.feature_names_in_)
        gb_confidence = boosting_model.predict_proba(feature_vector_selected)[0][list(boosting_model.classes_).index(diagnosis["illness"])]

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": round(gb_confidence, 2),
        })

    return refined_diagnoses

# Step 3: Final Confidence Adjustment with AdaBoost
def adaboost_ranking(possible_diagnoses, fact_base):
    user_symptoms = [s["name"].lower().strip() for s in fact_base["symptoms"]]
    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        feature_vector = pd.DataFrame(
            [[1 if symptom in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms
        )
        feature_vector_selected = align_features(feature_vector[selected_features], boosting_model.feature_names_in_)

        gb_predictions = boosting_model.predict_proba(feature_vector_selected)
        feature_vector_adaboost = pd.concat([feature_vector_selected, pd.DataFrame(gb_predictions, columns=[f"GB_{cls}" for cls in boosting_model.classes_])], axis=1)
        feature_vector_adaboost = align_features(feature_vector_adaboost, adaboost_model.feature_names_in_)

        ab_confidence = adaboost_model.predict_proba(feature_vector_adaboost)[0][list(adaboost_model.classes_).index(diagnosis["illness"])]

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": diagnosis["confidence_gb"],
            "confidence_ab": round(ab_confidence, 2),
        })

    return refined_diagnoses

@app.route("/diagnose", methods=["POST"])
def diagnose():
    fact_base = request.json
    possible_diagnoses = forward_chaining(fact_base)

    if not possible_diagnoses:
        return jsonify({"error": "No matching illnesses found. Try entering different symptoms."}), 400

    refined_diagnoses = gradient_boosting_ranking(possible_diagnoses, fact_base)
    final_diagnoses = adaboost_ranking(refined_diagnoses, fact_base)
    final_diagnoses.sort(key=lambda x: -x["confidence_ab"])

    return jsonify({"diagnosis": final_diagnoses[:5]})  # ✅ Return top 5 results

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
