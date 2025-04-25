import joblib
import numpy as np
import json
import pandas as pd
import os

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


# Placeholder follow-up question mapping
symptom_followups = {
    "vomiting": ["How long has your pet been vomiting?", "Is the vomiting severe?"],
    "diarrhea": ["How long has your pet had diarrhea?", "Is there blood in the stool?"],
    "coughing": ["How long has your pet been coughing?", "Is the cough dry or wet?"],
    "fever": ["Has your pet been lethargic?", "Is the fever high (above 39.5°C)?"],
}

def categorize_age(age):
    age = int(age)  # Ensure it's an integer
    if age <= 1:
        return "Puppy"
    elif 1 < age <= 7:
        return "Adult"
    else:
        return "Senior"

# Get User Input
def get_user_input():
    print("\n🔹 Enter pet owner details:")
    owner_name = input("➡ Owner's Name: ").strip()

    print("\n🔹 Enter pet details:")
    age_years = input("➡ Pet Age (in years): ").strip()
    age_range = categorize_age(age_years)  # ✅ Convert numeric age to category
    weight = input("➡ Pet Weight (kg): ").strip()
    height = input("➡ Pet Height (cm): ").strip()
    breed = input("➡ Pet Breed (or type 'Any'): ").strip().capitalize()
    medical_history = input("➡ Does your pet have any medical history? (Yes/No): ").strip().lower()
    previous_medications = input("➡ Has your pet taken any medication recently? (Yes/No): ").strip().lower()

    symptoms = []  # Store all user symptoms and their details

    while True:
        # Ask for the main symptom
        print("\n🔹 Enter the primary symptom troubling your pet:")
        main_symptom = input("➡ Symptom: ").strip().lower()
        symptom_details = {"name": main_symptom}

        # ✅ Collect follow-up answers, but they will not be used yet
        if main_symptom in symptom_followups:
            for question in symptom_followups[main_symptom]:
                answer = input(f"➡ {question} ").strip()
                symptom_details[question] = answer  # Store response but do nothing with it

        symptoms.append(symptom_details)

        # Ask if the user wants to add more symptoms
        more_symptoms = input("➡ Does your pet have any other symptoms? (Yes/No): ").strip().lower()
        if more_symptoms != "yes":
            break  # Stop collecting symptoms

    return {
        "owner": owner_name,
        "pet_info": {
            "age_years": age_years,  # Numeric value
            "age_range": age_range,  # Categorized value
            "weight": weight,
            "height": height,
            "breed": breed,
            "medical_history": medical_history,
            "previous_medications": previous_medications,
        },
        "symptoms": symptoms,  
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
    user_symptoms = [s["name"].lower().strip() for s in fact_base["symptoms"] if isinstance(s, dict) and "name" in s]
    pet_info = fact_base["pet_info"]

    possible_diagnoses = []

    for rule in load_knowledge_base():
        matched_symptoms = []
        total_weight = sum(symptom["weight"] for symptom in rule["symptoms"])
        matched_weight = 0  

        for symptom in rule["symptoms"]:
            symptom_name = symptom["name"].lower().strip()

            if symptom_name in user_symptoms:
                matched_weight += symptom["weight"]
                matched_symptoms.append(symptom_name)

        if matched_weight > 0:
            match_ratio = matched_weight / total_weight
            initial_confidence = rule["confidence"] * match_ratio  

            # ✅ Use `.get()` to avoid KeyError
            age_range = rule.get("age_range", "Any").lower()
            breed = rule.get("breed", "Any").lower()
            size = rule.get("size", "Any").lower()

            pet_age_range = pet_info.get("age_range", "Any").lower()
            pet_breed = pet_info.get("breed", "Any").lower()
            pet_size = pet_info.get("size", "Any").lower()

            # ✅ Default multipliers
            age_match = 1.0
            breed_match = 1.0
            size_match = 1.0

            # 🔹 Apply Age Filtering
            if age_range != "any" and age_range != pet_age_range:
                age_match = 0.85
            elif age_range == pet_age_range:
                age_match = 1.1

            # 🔹 Apply Breed Filtering
            if breed != "any" and breed != pet_breed:
                breed_match = 0.9
            elif breed == pet_breed:
                breed_match = 1.1

            # 🔹 Apply Size Filtering
            if size != "any" and size != pet_size:
                size_match = 0.9
            elif size == pet_size:
                size_match = 1.1

            # ✅ Adjust confidence based on pet details
            final_confidence = round(initial_confidence * age_match * breed_match * size_match, 2)

            possible_diagnoses.append({
                "illness": rule["illness"],
                "matched_symptoms": matched_symptoms,
                "confidence_fc": final_confidence,
            })

    return possible_diagnoses

# Step 2: Gradient Boosting - Illness Ranking Optimization
def gradient_boosting_ranking(possible_diagnoses, fact_base):
    # ✅ Ensure `user_symptoms` is always a list of strings
    if isinstance(fact_base["symptoms"], list):
        if isinstance(fact_base["symptoms"][0], dict):
            user_symptoms = [s["name"].lower().strip() for s in fact_base["symptoms"] if "name" in s]
        else:
            user_symptoms = [s.lower().strip() for s in fact_base["symptoms"]]
    else:
        user_symptoms = []

    pet_info = fact_base["pet_info"]
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

        # ✅ Retrieve illness rule safely
        rule = next((r for r in load_knowledge_base() if r["illness"] == diagnosis["illness"]), None)

        age_match = 1.0
        breed_match = 1.0
        size_match = 1.0

        if rule:
            age_range = rule.get("age_range", "Any").lower()
            breed = rule.get("breed", "Any").lower()
            size = rule.get("size", "Any").lower()

            pet_age_range = pet_info.get("age_range", "Any").lower()
            pet_breed = pet_info.get("breed", "Any").lower()
            pet_size = pet_info.get("size", "Any").lower()


            if age_range != "any" and age_range != pet_age_range:
                age_match = 0.85
            elif age_range == pet_age_range:
                age_match = 1.1

            if breed != "any" and breed != pet_breed:
                breed_match = 0.9
            elif breed == pet_breed:
                breed_match = 1.1

            if size != "any" and size != pet_size:
                size_match = 0.9
            elif size == pet_size:
                size_match = 1.1

        final_confidence_gb = round(gb_confidence * age_match * breed_match * size_match, 2)

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": final_confidence_gb,
        })

    return refined_diagnoses

# Step 3: Final Confidence Adjustment with AdaBoost
def adaboost_ranking(possible_diagnoses, fact_base):
    user_symptoms = [s["name"].lower().strip() for s in fact_base["symptoms"] if isinstance(s, dict) and "name" in s]
    pet_info = fact_base["pet_info"]

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

        ab_confidence = raw_predictions[0][illness_index] * (1 + (symptom_match_ratio * 0.5))

        # ✅ Retrieve illness rule safely
        rule = next((r for r in load_knowledge_base() if r["illness"] == diagnosis["illness"]), None)

        age_match = 1.0
        breed_match = 1.0
        size_match = 1.0

        if rule:
            age_range = rule.get("age_range", "Any").lower()
            breed = rule.get("breed", "Any").lower()
            size = rule.get("size", "Any").lower()

            pet_age_range = pet_info.get("age_range", "Any").lower()
            pet_breed = pet_info.get("breed", "Any").lower()
            pet_size = pet_info.get("size", "Any").lower()

            if age_range != "any" and age_range != pet_age_range:
                age_match = 0.85  # Penalize mismatches
            elif age_range == pet_age_range:
                age_match = 1.2  # ✅ Increase from 1.1 → 1.2 to give stronger preference

            if breed != "any" and breed != pet_breed:
                breed_match = 0.9
            elif breed == pet_breed:
                breed_match = 1.1

            if size != "any" and size != pet_size:
                size_match = 0.9
            elif size == pet_size:
                size_match = 1.1

        final_confidence_ab = round(ab_confidence * age_match * breed_match * size_match, 2)

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": diagnosis["confidence_gb"],
            "confidence_ab": final_confidence_ab,
        })

    return refined_diagnoses

# Function to save fact base results
def save_fact_base(fact_base):
    os.makedirs(os.path.dirname(FACT_BASE_PATH), exist_ok=True)  # Ensure directory exists
    with open(FACT_BASE_PATH, "w") as fb_file:
        json.dump(fact_base, fb_file, indent=4)
    print("\n✅ Results saved to fact_base.json!")

# Main Execution Flow with Results Saving
def run_diagnosis():
    fact_base = get_user_input()  # Get user input

    possible_diagnoses = forward_chaining(fact_base)

    if not possible_diagnoses:
        print("\n❌ No matching illnesses found. Try entering different symptoms.")
        return

    refined_diagnoses = gradient_boosting_ranking(possible_diagnoses, fact_base)
    final_diagnoses = adaboost_ranking(refined_diagnoses, fact_base)

    # ✅ Sort illnesses by highest confidence
    final_diagnoses.sort(key=lambda x: -x["confidence_ab"])

    # ✅ Limit the displayed results (Change to top 3 or top 5 as needed)
    top_results = final_diagnoses[:5]  # Change to `[:3]` for top 3 only

    # ✅ Load illness info from JSON
    illness_info = load_illness_info()

    # ✅ Store all results but display only the top ones
    fact_base["possible_diagnosis"] = final_diagnoses  # Save full results
    save_fact_base(fact_base)  # ✅ Save results to `fact_base.json`

    # ✅ Print Pet Info Summary
    print("\n📌 **Pet Information Summary:**")
    print(f"🔹 Age: {fact_base['pet_info']['age_years']} years")
    print(f"🔹 Weight: {fact_base['pet_info']['weight']} kg")
    print(f"🔹 Height: {fact_base['pet_info']['height']} cm")
    print(f"🔹 Breed: {fact_base['pet_info']['breed']}")

    # ✅ Print Symptoms Summary
    print("\n🩺 **Symptoms Provided:**")
    for symptom in fact_base["symptoms"]:
        print(f"🔹 {symptom['name'].capitalize()}")

    # ✅ Display only the top diagnoses
    print("\n🩺 **Final Diagnoses (Forward Chaining + Boosting + AdaBoost):**")
    for diagnosis in top_results:
        print(f"🔹 {diagnosis['illness']} (Confidence: {diagnosis['confidence_ab']})")

        # ✅ Retrieve illness rule safely
        rule = next((r for r in load_knowledge_base() if r["illness"] == diagnosis["illness"]), None)

        if rule:
            # ✅ Find matched and missing symptoms
            matched_symptoms = set(diagnosis["matched_symptoms"])
            illness_symptoms = set([s["name"].lower() for s in rule["symptoms"]])
            missing_symptoms = illness_symptoms - matched_symptoms

            # ✅ Display illness reasoning
            print(f"   🔹 **Why this illness?**")
            print(f"      - Matched Symptoms: {', '.join(diagnosis['matched_symptoms'])}")

            # ✅ Get common symptoms from illness_info or fallback to knowledge_base
            illness_common_symptoms = illness_info.get(diagnosis['illness'], {}).get("common_symptoms", [])

            # 🔹 If `illness_common_symptoms` is missing, get symptoms from knowledge_base
            if not illness_common_symptoms:
                rule = next((r for r in load_knowledge_base() if r["illness"] == diagnosis["illness"]), None)
                if rule:
                    illness_common_symptoms = [s["name"] for s in rule["symptoms"]]  # Fallback to knowledge_base

            # ✅ Display only the missing ones
            missing_symptoms_display = ', '.join(set(illness_common_symptoms) - set(diagnosis['matched_symptoms']))
            print(f"      - Common Symptoms Not Reported: {missing_symptoms_display if missing_symptoms_display else 'None ✅'}")


            # ✅ Age/Breed/Size Influence
            age_range = rule.get("age_range", "Any").lower()
            breed = rule.get("breed", "Any").lower()
            size = rule.get("size", "Any").lower()
            pet_age_range = fact_base["pet_info"].get("age_range", "Any").lower()
            pet_breed = fact_base["pet_info"].get("breed", "Any").lower()
            pet_size = fact_base["pet_info"].get("size", "Any").lower()

            if age_range != "any" and age_range != pet_age_range:
                print(f"      - ⚠️ This illness is usually found in {age_range.capitalize()} dogs, but your pet is {pet_age_range}.")

            if breed != "any" and breed != pet_breed:
                print(f"      - ⚠️ This illness is more common in {breed} breeds, but your pet is a {pet_breed}.")

            if size != "any" and size != pet_size:
                print(f"      - ⚠️ This illness is more typical in {size} dogs, but your pet is {pet_size}.")

        # ✅ Fetch illness details from JSON
        illness_details = illness_info.get(diagnosis["illness"], {})
        description = illness_details.get("description", "No description available.")
        severity = illness_details.get("severity", "Unknown")
        treatment = illness_details.get("treatment", "No treatment guidelines provided.")

        # ✅ Display Illness Details
        print(f"\n   📌 **About {diagnosis['illness']}**")
        print(f"      - {description}")
        print(f"      - **Severity:** {severity}")
        print(f"      - **Recommended Action:** {treatment}\n")

    print("\n✅ Diagnosis with full illness details included!")

# Run the system
if __name__ == "__main__":
    run_diagnosis()