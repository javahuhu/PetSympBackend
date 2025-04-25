import json
import joblib
import numpy as np


# Load Knowledge Base (Rules)
def load_knowledge_base():
    with open("data/knowledge_base.json", "r") as kb_file:
        return json.load(kb_file)["rules"]


# Save Updated Fact Base
def save_fact_base(fact_base):
    with open("data/fact_base.json", "w") as fb_file:
        json.dump(fact_base, fb_file, indent=4)


# Get User Input (CLI-based)
def get_user_input():
    print("\n🔹 Enter your pet's symptoms (comma-separated, e.g., Vomiting, Diarrhea):")
    user_symptoms = input("➡ Symptoms: ").strip().split(",")
    user_symptoms = [symptom.strip().capitalize() for symptom in user_symptoms]

    # Get pet details
    age_range = input("➡ Pet Age Range (Puppy, Adult, Senior): ").strip().capitalize()
    breed = input("➡ Pet Breed (or type 'Any'): ").strip().capitalize()
    size = input("➡ Pet Size (Small, Medium, Large): ").strip().capitalize()

    # Update fact base
    fact_base = {
        "pet_info": {"age_range": age_range, "breed": breed, "size": size},
        "symptoms": user_symptoms,
    }
    save_fact_base(fact_base)
    print("\n✅ Fact Base Updated! Now running Forward Chaining...\n")
    return fact_base


# Load trained Boosting model and label encoder
gb_model = joblib.load("gradient_boosting_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")  # Load saved feature order

print("✅ Forward Chaining + Boosting Model Loaded Successfully!")


# Forward Chaining Algorithm with Boosting Integration
def forward_chaining():
    knowledge_base = load_knowledge_base()  # Load illness rules
    fact_base = get_user_input()  # Get real-time user input
    user_symptoms = fact_base["symptoms"]
    pet_info = fact_base["pet_info"]

    possible_diagnoses = []

    for rule in knowledge_base:
        matched_symptoms = []
        total_weight = 0
        matched_weight = 0
        total_priority = 0
        matched_priority = 0

        for symptom in rule["symptoms"]:
            total_weight += symptom["weight"]  # Sum of all possible symptom weights
            total_priority += symptom["priority"]  # Sum of all possible priorities
            if symptom["name"] in user_symptoms:
                matched_symptoms.append(symptom)
                matched_weight += symptom["weight"]  # Sum of matched symptom weights
                matched_priority += symptom["priority"]  # Sum of matched priorities

        if matched_symptoms:  # If at least one symptom matches
            match_percentage = matched_weight / total_weight  # Weighted match ratio
            priority_percentage = (
                matched_priority / total_priority
            )  # Priority match ratio
            illness_confidence = rule[
                "confidence"
            ]  # Base confidence from knowledge base

            # Define severity multipliers
            severity_weight = {"Low": 0.9, "Medium": 1.0, "High": 1.3, "Critical": 1.5}

            # Apply severity impact during confidence calculation
            severity_adjustment = 1.0
            for symptom in matched_symptoms:
                severity_adjustment *= severity_weight.get(
                    symptom.get("severity", "Medium"), 1.0
                )

            # Final confidence with severity impact
            final_confidence = round(
                illness_confidence
                * (
                    matched_weight
                    / sum(symptom["weight"] for symptom in matched_symptoms)
                )
                * severity_adjustment,
                2,
            )
            # Get the number of symptoms that matched
            symptom_match_count = len(matched_symptoms)

            # Penalize illnesses that match very few symptoms
            if symptom_match_count == 1:  # If only one symptom matched
                final_confidence *= 0.75  # Reduce confidence by 25%
            elif symptom_match_count == 2:
                final_confidence *= 0.9  # Reduce confidence by 10%

            # Adjust priority based on symptoms that matched
            final_priority = round((priority_percentage * 1.5) + illness_confidence, 2)

            # Incorporate Pet Characteristics into the Ranking
            age_match = (
                1.3
                if rule["age_range"] == pet_info["age_range"]
                else (1.0 if rule["age_range"] == "Any" else 0.8)
            )
            breed_match = (
                1.2
                if rule["breed"] == pet_info["breed"]
                else (1.0 if rule["breed"] == "Any" else 0.9)
            )
            size_match = (
                1.2
                if rule["size"] == pet_info["size"]
                else (1.0 if rule["size"] == "Any" else 0.9)
            )

            pet_match_factor = round(
                (age_match * 1.5 + breed_match + size_match) / 3, 2
            )

            # Apply the pet match factor to confidence and priority
            final_confidence = round(final_confidence * pet_match_factor, 2)
            final_priority = round(final_priority * pet_match_factor, 2)

            possible_diagnoses.append(
                {
                    "illness": rule["illness"],
                    "matched_symptoms": [s["name"] for s in matched_symptoms],
                    "confidence": final_confidence,
                    "priority": final_priority,
                }
            )

    # If multiple illnesses match, use Boosting to rank them
    if len(possible_diagnoses) > 1:
        return boosting_ranking(fact_base, possible_diagnoses)

    # If only one illness matches, return it
    if len(possible_diagnoses) == 1:
        return possible_diagnoses[0]["illness"]

    return "No illness matched."


# Boosting Model Integration for Ranking
def boosting_ranking(fact_base, illnesses):
    """
    Uses the trained Boosting model to rank potential illnesses based on confidence scores.
    """
    user_symptoms = fact_base["symptoms"]

    # Create a feature vector matching the dataset structure
    symptom_vector = np.zeros(len(feature_names))

    # Set symptoms to 1 if they are in user input
    for i, symptom in enumerate(feature_names):
        if symptom in user_symptoms:
            symptom_vector[i] = 1

    # Predict illness using Boosting model
    predicted_illness_index = gb_model.predict([symptom_vector])[0]

    # Convert index back to illness name
    predicted_illness = label_encoder.inverse_transform([predicted_illness_index])[0]

    # Rank illnesses (Boosting model takes final decision)
    ranked_illnesses = sorted(illnesses, key=lambda x: x["confidence"], reverse=True)

    print("\n🩺 **Diagnosis Results:**")
    print(f"🔹 ML Ranked Illness: {predicted_illness}")
    for diagnosis in ranked_illnesses:
        print(
            f"🔹 {diagnosis['illness']} (Confidence: {diagnosis['confidence']}, Priority: {diagnosis['priority']})"
        )

    return f"ML Final Diagnosis: {predicted_illness}"


# Run the algorithm
if __name__ == "__main__":
    forward_chaining()
