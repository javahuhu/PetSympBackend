import joblib
import numpy as np
import json
import pandas as pd
import os

# 📌 File Paths
FACT_BASE_PATH = "data/fact_base.json"
KNOWLEDGE_BASE_PATH = "data/updated_knowledge_base_v2_fixed.json"
ILLNESS_INFO_PATH = "data/illness_info.json"
FOLLOWUP_QUESTIONS_PATH = "data/updated_follow_up_questions_tuned.json"
DATASET_PATH = "canine_illness_dataset.csv"

# 📌 Load JSON Files
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# 📌 Load Data
knowledge_base = load_json(KNOWLEDGE_BASE_PATH)["rules"]
illness_info = load_json(ILLNESS_INFO_PATH)
symptom_followups = load_json(FOLLOWUP_QUESTIONS_PATH)

# 📌 Load Machine Learning Models
boosting_model = joblib.load("boosting_model.pkl")
adaboost_model = joblib.load("adaboost_model.pkl")
selected_features = joblib.load("selected_features.pkl")

# Load Knowledge Base (Rules)
def load_knowledge_base():
    return knowledge_base  # Uses the preloaded knowledge base

# 📌 Load Illness Information Database
def load_illness_info():
    """Loads illness information from JSON file."""
    with open(ILLNESS_INFO_PATH, "r") as illness_file:
        return json.load(illness_file)

# 📌 Load Dataset & Features
df = pd.read_csv(DATASET_PATH)
all_symptoms = [col for col in df.columns if col not in ["Test Case ID", "Illness"]]

# 📌 Categorize Pet Age
def categorize_age(age):
    """Convert numeric age to category."""
    try:
        age = int(age)  # Ensure it's an integer
    except ValueError:
        return "Unknown"  # Handle non-integer inputs safely

    if age <= 1:
        return "Puppy"
    elif 1 < age <= 7:
        return "Adult"
    else:
        return "Senior"

# 📌 Get User Input
def get_user_input():
    """Collect user input for pet details and symptoms."""
    print("\n🔹 Enter pet owner details:")
    owner_name = input("➡ Owner's Name: ").strip()

    print("\n🔹 Enter pet details:")
    age_years = input("➡ Pet Age (in years): ").strip()
    age_range = categorize_age(age_years)  # Convert numeric age to category
    weight = input("➡ Pet Weight (kg): ").strip()
    height = input("➡ Pet Height (cm): ").strip()
    breed = input("➡ Pet Breed (or type 'Any'): ").strip().capitalize()
    medical_history = input("➡ Does your pet have any medical history? (Yes/No): ").strip().lower()
    previous_medications = input("➡ Has your pet taken any medication recently? (Yes/No): ").strip().lower()

    symptoms = []
    user_answers = {}  # ✅ Always initialize `user_answers`

    while True:
        print("\n🔹 Enter the primary symptom troubling your pet:")
        main_symptom = input("➡ Symptom: ").strip().lower()
        symptom_details = {"name": main_symptom}

        # ✅ Check if symptom has follow-up questions
        if main_symptom in symptom_followups:
            user_answers[main_symptom] = {}  # ✅ Initialize answers for this symptom

            for question in symptom_followups[main_symptom]["questions"]:
                answer = input(f"➡ {question} ").strip().lower()
                user_answers[main_symptom][question] = answer  # ✅ Store response properly

        symptoms.append(symptom_details)

        # ✅ Ask if there are more symptoms
        if input("➡ More symptoms? (Yes/No): ").strip().lower() != "yes":
            break

    return {
        "owner": owner_name,
        "pet_info": {
            "age_years": age_years,
            "age_range": age_range,
            "weight": weight,
            "height": height,
            "breed": breed,
            "medical_history": medical_history,
            "previous_medications": previous_medications,
        },
        "symptoms": symptoms,
        "user_answers": user_answers,  # ✅ Ensure `user_answers` is always returned
    }

# 📌 Ensure all expected features exist in the dataset (missing ones should be filled with 0)
def align_features(feature_vector, expected_features):
    """Align feature vector with expected features to ensure consistency."""
    feature_vector = feature_vector.copy()  # Prevents SettingWithCopyWarning

    # ✅ Add missing features as zeros
    for feature in expected_features:
        if feature not in feature_vector.columns:
            feature_vector.loc[:, feature] = 0  

    # ✅ Reorder columns to match training feature order
    return feature_vector[expected_features]

def adjust_confidence_with_followups(confidence, symptom_details, illness_name, user_answers):
    """Modify confidence scores based on user follow-up answers and KB expectations."""
    illness_rule = next((r for r in knowledge_base if r["illness"] == illness_name), None)

    if not illness_rule:
        return confidence  # No rule found, return original confidence

    total_multiplier = 1.0  # ✅ Start with a neutral multiplier

    for symptom in symptom_details:
        symptom_name = symptom["name"].lower()  # Ensure case-insensitive matching

        if symptom_name in symptom_followups and illness_rule:
            expected_symptoms = {s["name"].lower(): s for s in illness_rule["symptoms"]}

            if symptom_name in expected_symptoms:
                expected_data = expected_symptoms[symptom_name]
                expected_duration = expected_data.get("duration_range", "Any")
                expected_severity = expected_data.get("severity", "Any")
                expected_subtype = expected_data.get("subtype", "Any")

                # ✅ Convert expected_subtype into a **list** (for proper comparison)
                expected_subtypes = [s.strip().lower() for s in expected_subtype.split(",") if s.strip()]

                # ✅ Retrieve user responses safely
                user_response = user_answers.get(symptom_name, {})
                user_duration = user_response.get(f"How long has your pet had {symptom_name}?", None)
                user_severity = user_response.get(f"Is the {symptom_name} Mild, Moderate, or Severe?", None)

                # ✅ Retrieve subtype dynamically based on symptom
                user_subtype_key = next(
                    (q for q in symptom_followups[symptom_name]["questions"] if "Dry or Wet" in q or "Watery or Bloody" in q or "Mild or Severe" in q),
                    None
                )
                user_subtype = user_response.get(user_subtype_key, None) if user_subtype_key else None

                # ✅ NEW FIX: If user_subtype is **None**, check if severity response matches subtype
                if not user_subtype and user_severity and user_severity.lower() in expected_subtypes:
                    user_subtype = user_severity  # ✅ Assign severity response as subtype

                # ✅ Convert user_subtype into lowercase string for comparison
                user_subtype_clean = user_subtype.lower().strip() if user_subtype else None

                # ✅ Retrieve impact values safely
                impact_values = symptom_followups.get(symptom_name, {}).get("impact", {})

                # 🔍 DEBUGGING - Check what is being used
                print(f"\n🔍 DEBUG: Checking {illness_name} for symptom: {symptom_name}")
                print(f"   - KB Expected Severity: {expected_severity}")  
                print(f"   - KB Expected Subtype(s): {expected_subtypes}")  
                print(f"   - User Severity Input: {user_severity}")
                print(f"   - User Subtype Input (FIXED): {user_subtype_clean}")
                print(f"   - Subtype Match? {'✅ YES' if user_subtype_clean and user_subtype_clean in expected_subtypes else '❌ NO'}")
                print(f"   - Expected Duration: {expected_duration}")
                print(f"   - User Duration: {user_duration}")

                # ✅ Ensure impact retrieval is case-insensitive and provides a default
                severity_impact = impact_values.get(user_severity.lower(), 1.2) if user_severity else 1.2
                subtype_impact = impact_values.get(user_subtype_clean, 1.2) if user_subtype_clean else 1.2
                duration_impact = impact_values.get(user_duration.lower(), 1.2) if user_duration else 1.2

                kb_match_bonus = 1.0  # Default no change

                # ✅ Apply KB Matching Bonus or Penalty for Severity & Duration
                if user_severity and expected_severity != "Any":
                    if user_severity.lower() == expected_severity.lower():
                        kb_match_bonus *= 1.02  # ✅ Small boost if severity matches KB
                    else:
                        kb_match_bonus *= 0.95  # ❌ Small penalty if severity mismatches

                if user_duration and expected_duration != "Any":
                    if user_duration in expected_duration:
                        kb_match_bonus *= 1.03  # ✅ Small boost if duration matches KB
                    else:
                        kb_match_bonus *= 0.95  # ❌ Small penalty if duration mismatches

                # ✅ NEW: Apply KB Matching Bonus or Penalty for Subtype
                if user_subtype_clean and expected_subtypes != ["any"]:  # ✅ Ensure "Any" doesn't block valid matches
                    if user_subtype_clean in expected_subtypes:
                        kb_match_bonus *= 1.08  # ✅ Small boost if subtype matches KB
                    else:
                        kb_match_bonus *= 0.9  # ❌ Small penalty if subtype mismatches

                # ✅ Apply all multipliers
                total_multiplier *= severity_impact * subtype_impact * duration_impact * kb_match_bonus

    confidence *= total_multiplier
    print(f"🔍 Final Adjusted Confidence for {illness_name}: {confidence}\n")  
    print(f"🔍 Final Multipliers Applied:")
    print(f"   - Duration Impact: {duration_impact}")
    print(f"   - Severity Impact: {severity_impact}")
    print(f"   - Subtype Impact: {subtype_impact}")
    print(f"   - KB Match Bonus: {kb_match_bonus}")
    print(f"   - Total Multiplier: {total_multiplier}")

    return round(confidence, 2)

# 📌 Step 1: Forward Chaining - Initial Illness Matching
def forward_chaining(fact_base):
    """Matches user symptoms with knowledge base rules using a weighted confidence approach."""
    
    user_symptoms = [s["name"].lower().strip() for s in fact_base["symptoms"] if isinstance(s, dict) and "name" in s]
    pet_info = fact_base["pet_info"]
    user_answers = fact_base["user_answers"]  # ✅ Include user responses for follow-ups

    possible_diagnoses = []

    for rule in load_knowledge_base():
        matched_symptoms = []
        total_weight = sum(symptom["weight"] for symptom in rule["symptoms"])
        matched_weight = sum(
            symptom["weight"] for symptom in rule["symptoms"] if symptom["name"].lower().strip() in user_symptoms
        )

        if matched_weight > 0:
            match_ratio = matched_weight / total_weight
            initial_confidence = round(rule["confidence"] * match_ratio, 2)

            # ✅ Retrieve rule-based constraints with safe defaults
            age_range = rule.get("age_range", "Any").lower()
            breed = rule.get("breed", "Any").lower()
            size = rule.get("size", "Any").lower()

            pet_age_range = pet_info.get("age_range", "Any").lower()
            pet_breed = pet_info.get("breed", "Any").lower()
            pet_size = pet_info.get("size", "Any").lower()

            # ✅ Default multipliers
            age_match = 1.0 if age_range == "any" else (1.1 if age_range == pet_age_range else 0.85)
            breed_match = 1.0 if breed == "any" else (1.1 if breed == pet_breed else 0.9)
            size_match = 1.0 if size == "any" else (1.1 if size == pet_size else 0.9)

            # ✅ Apply follow-up question adjustments before finalizing confidence
            final_confidence = adjust_confidence_with_followups(
                round(initial_confidence * age_match * breed_match * size_match, 2),
                fact_base["symptoms"],
                rule["illness"],
                user_answers  # ✅ Pass user answers for dynamic adjustments
            )

            possible_diagnoses.append({
                "illness": rule["illness"],
                "matched_symptoms": [s["name"].lower().strip() for s in rule["symptoms"] if s["name"].lower().strip() in user_symptoms],
                "confidence_fc": final_confidence,
            })

    return possible_diagnoses

# 📌 Step 2: Gradient Boosting - Illness Ranking Optimization
def gradient_boosting_ranking(possible_diagnoses, fact_base):
    """Uses Gradient Boosting to refine illness ranking based on symptom match and model prediction."""
    
    # ✅ Ensure `user_symptoms` is a properly formatted list
    user_symptoms = [
        s["name"].lower().strip() for s in fact_base["symptoms"] 
        if isinstance(s, dict) and "name" in s
    ] if isinstance(fact_base["symptoms"], list) else []

    pet_info = fact_base["pet_info"]
    user_answers = fact_base["user_answers"]  # ✅ Include user responses for follow-ups

    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # ✅ Create feature vector for ML model
        feature_vector = pd.DataFrame(
            [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms
        )

        feature_vector_selected = align_features(
            feature_vector[selected_features], boosting_model.feature_names_in_
        )

        # ✅ Get Gradient Boosting prediction
        raw_predictions = boosting_model.predict_proba(feature_vector_selected)
        illness_index = list(boosting_model.classes_).index(diagnosis["illness"])
        gb_confidence = raw_predictions[0][illness_index]

        # ✅ Retrieve illness rule safely
        rule = next((r for r in load_knowledge_base() if r["illness"] == diagnosis["illness"]), None)

        # ✅ Default multipliers
        age_match = breed_match = size_match = 1.0

        if rule:
            age_match = 1.1 if rule.get("age_range", "Any").lower() == pet_info.get("age_range", "Any").lower() else 0.85
            breed_match = 1.1 if rule.get("breed", "Any").lower() == pet_info.get("breed", "Any").lower() else 0.9
            size_match = 1.1 if rule.get("size", "Any").lower() == pet_info.get("size", "Any").lower() else 0.9

        # ✅ Apply follow-up question adjustments before finalizing confidence
        final_confidence_gb = adjust_confidence_with_followups(
            round(gb_confidence * age_match * breed_match * size_match, 2),
            fact_base["symptoms"],
            diagnosis["illness"],
            user_answers  # ✅ Pass user answers for dynamic adjustments
        )

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": final_confidence_gb,
        })

    return refined_diagnoses

# 📌 Step 3: Final Confidence Adjustment with AdaBoost
def adaboost_ranking(possible_diagnoses, fact_base):
    """Uses AdaBoost to refine final illness ranking based on symptom match and ML confidence adjustments."""
    
    # ✅ Ensure `user_symptoms` is a properly formatted list
    user_symptoms = [
        s["name"].lower().strip() for s in fact_base["symptoms"]
        if isinstance(s, dict) and "name" in s
    ] if isinstance(fact_base["symptoms"], list) else []

    pet_info = fact_base["pet_info"]
    user_answers = fact_base["user_answers"]  # ✅ Include user responses for follow-ups

    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # ✅ Create feature vector for ML model
        feature_vector = pd.DataFrame(
            [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms
        )

        feature_vector_selected = align_features(
            feature_vector[selected_features], boosting_model.feature_names_in_
        )

        # ✅ Get Gradient Boosting predictions
        gb_predictions = boosting_model.predict_proba(feature_vector_selected)
        gb_predictions_df = pd.DataFrame(
            gb_predictions, columns=[f"GB_{cls}" for cls in boosting_model.classes_]
        )

        # ✅ Prepare feature vector for AdaBoost
        feature_vector_adaboost = pd.concat([feature_vector_selected, gb_predictions_df], axis=1)
        feature_vector_adaboost = align_features(feature_vector_adaboost, adaboost_model.feature_names_in_)

        # ✅ Get AdaBoost predictions
        raw_predictions = adaboost_model.predict_proba(feature_vector_adaboost)
        illness_index = list(adaboost_model.classes_).index(diagnosis["illness"])
        symptom_match_ratio = len(diagnosis.get("matched_symptoms", [])) / max(len(user_symptoms), 1)

        # ✅ Calculate AdaBoost confidence with symptom influence
        ab_confidence = raw_predictions[0][illness_index] * (1 + (symptom_match_ratio * 0.5))

        # ✅ Retrieve illness rule safely
        rule = next((r for r in load_knowledge_base() if r["illness"] == diagnosis["illness"]), None)

        # ✅ Default multipliers
        age_match = breed_match = size_match = 1.0

        if rule:
            age_match = 1.2 if rule.get("age_range", "Any").lower() == pet_info.get("age_range", "Any").lower() else 0.85
            breed_match = 1.1 if rule.get("breed", "Any").lower() == pet_info.get("breed", "Any").lower() else 0.9
            size_match = 1.1 if rule.get("size", "Any").lower() == pet_info.get("size", "Any").lower() else 0.9

        # ✅ Apply follow-up question adjustments before finalizing confidence
        final_confidence_ab = adjust_confidence_with_followups(
            round(ab_confidence * age_match * breed_match * size_match, 2),
            fact_base["symptoms"],
            diagnosis["illness"],
            user_answers  # ✅ Pass user answers for dynamic adjustments
        )

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": diagnosis["confidence_gb"],
            "confidence_ab": final_confidence_ab,
        })

    return refined_diagnoses

# 📌 Function to Save Fact Base Results
def save_fact_base(fact_base):
    """Ensures the directory exists and saves the fact base to a JSON file."""
    os.makedirs(os.path.dirname(FACT_BASE_PATH), exist_ok=True)
    with open(FACT_BASE_PATH, "w") as fb_file:
        json.dump(fact_base, fb_file, indent=4)
    print("\n✅ Results saved to fact_base.json!")


# 📌 Main Execution Flow with Results Saving
def run_diagnosis():
    """Executes the diagnosis process using Forward Chaining, Boosting, and AdaBoost models."""
    
    # ✅ Get user input
    fact_base = get_user_input()
    
    # ✅ Step 1: Forward Chaining - Initial Symptom Matching
    possible_diagnoses = forward_chaining(fact_base)
    if not possible_diagnoses:
        print("\n❌ No matching illnesses found. Try entering different symptoms.")
        return

    # ✅ Step 2: Gradient Boosting - Illness Ranking Optimization
    refined_diagnoses = gradient_boosting_ranking(possible_diagnoses, fact_base)

    # ✅ Step 3: AdaBoost - Final Confidence Adjustment
    final_diagnoses = adaboost_ranking(refined_diagnoses, fact_base)

    # ✅ Sort illnesses by highest confidence
    final_diagnoses.sort(key=lambda x: -x["confidence_ab"])

    # ✅ Display only the top 5 diagnoses (Adjust to `[:3]` for top 3 results)
    top_results = final_diagnoses[:5]

    # ✅ Store all results but display only the top ones
    fact_base["possible_diagnosis"] = final_diagnoses
    save_fact_base(fact_base)  # ✅ Save results to `fact_base.json`

    # ✅ Load illness info from JSON
    illness_info = load_illness_info()

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

    # ✅ Display the top diagnoses
    print("\n🩺 **Final Diagnoses (Forward Chaining + Boosting + AdaBoost):**")
    for diagnosis in top_results:
        print(f"🔹 {diagnosis['illness']} (Confidence: {diagnosis['confidence_ab']})")

        # ✅ Retrieve illness rule safely
        rule = next((r for r in load_knowledge_base() if r["illness"] == diagnosis["illness"]), None)

        if rule:
            matched_symptoms = set(diagnosis["matched_symptoms"])
            illness_symptoms = {s["name"].lower() for s in rule["symptoms"]}
            missing_symptoms = illness_symptoms - matched_symptoms

            # ✅ Display illness reasoning
            print(f"   🔹 **Why this illness?**")
            print(f"      - Matched Symptoms: {', '.join(diagnosis['matched_symptoms'])}")

            # ✅ Get common symptoms from illness_info or fallback to knowledge_base
            illness_common_symptoms = illness_info.get(diagnosis['illness'], {}).get("common_symptoms", [])
            
            if not illness_common_symptoms:  # Fallback if no common symptoms in illness_info.json
                illness_common_symptoms = [s["name"] for s in rule["symptoms"]]

            # ✅ Display only the missing ones
            missing_symptoms_display = ', '.join(set(illness_common_symptoms) - matched_symptoms)
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


# 📌 Run the system
if __name__ == "__main__":
    run_diagnosis()
