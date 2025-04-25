import json


# Load Knowledge Base (Rules)
def load_knowledge_base():
    with open("data/new_knowledge_base.json", "r") as kb_file:
        return json.load(kb_file)["rules"]


# Save Updated Fact Base
def save_fact_base(fact_base):
    with open("data/fact_base.json", "w") as fb_file:
        json.dump(fact_base, fb_file, indent=4)


# Get User Input (CLI-based)
def get_user_input():
    print(
        "\n🔹 Enter your pet's symptoms (comma-separated, e.g., Vomiting, Loss of Appetite):"
    )
    user_symptoms = input("➡ Symptoms: ").strip().split(",")

    # Normalize symptoms to lowercase and strip spaces
    user_symptoms = [symptom.strip().lower() for symptom in user_symptoms]

    # Get pet details
    age_range = input("➡ Pet Age Range (Puppy, Adult, Senior): ").strip().capitalize()
    breed = input("➡ Pet Breed (or type 'Any'): ").strip().capitalize()
    size = input("➡ Pet Size (Small, Medium, Large): ").strip().capitalize()

    fact_base = {
        "pet_info": {"age_range": age_range, "breed": breed, "size": size},
        "symptoms": user_symptoms,
    }
    save_fact_base(fact_base)
    print("\n✅ Fact Base Updated! Now running Forward Chaining...\n")
    return fact_base


# Forward Chaining Algorithm with Improved Weighting and Pet Characteristics
def forward_chaining():
    new_knowledge_base = load_knowledge_base()  # Load illness rules
    fact_base = get_user_input()  # Get real-time user input
    user_symptoms = fact_base["symptoms"]
    pet_info = fact_base["pet_info"]

    possible_diagnoses = []

    for rule in new_knowledge_base:
        matched_symptoms = []
        total_weight = 0
        matched_weight = 0
        total_priority = 0
        matched_priority = 0

        for symptom in rule["symptoms"]:
            total_weight += symptom["weight"]  # Sum of all possible symptom weights
            total_priority += symptom["priority"]  # Sum of all possible priorities
            if (
                symptom["name"].lower() in user_symptoms
            ):  # Ensure matching is case-insensitive
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

            # Calculate match ratio to prioritize illnesses with more symptom matches
            match_ratio = len(matched_symptoms) / len(rule["symptoms"])

            # Adjust confidence to balance match percentage and total symptoms matched
            symptom_match_ratio = len(matched_symptoms) / len(rule["symptoms"])  # Percentage of illness symptoms matched
            symptom_coverage_ratio = len(matched_symptoms) / len(user_symptoms)  # Percentage of user symptoms covered

            # Balance confidence using both symptom match ratio and absolute count of symptoms matched
            final_confidence = round(
                illness_confidence
                * ((symptom_match_ratio + symptom_coverage_ratio) / 2)  # Weigh both ratios equally
                * (
                    matched_weight
                    / max(sum(symptom["weight"] for symptom in matched_symptoms), 1)
                )
                * severity_adjustment,
                2,
            )


            # Scale confidence boost based on symptom match percentage, adjusted for illness size
            match_percentage = len(matched_symptoms) / len(rule["symptoms"])

            # Dynamic threshold ensures illnesses with many symptoms don’t require 100% to get boosted
            dynamic_threshold = max(0.6, 1 - (0.05 * len(rule["symptoms"])))

            if match_percentage >= dynamic_threshold:
                final_confidence *= 1 + (
                    0.2 * match_percentage
                )  # Scale boost dynamically

            # Get the number of symptoms that matched
            symptom_match_count = len(matched_symptoms)

            # Penalize illnesses that match very few symptoms
            if symptom_match_count == 1:
                final_confidence *= 0.75
            elif symptom_match_count == 2:
                final_confidence *= 0.9

            # Adjust priority based on symptoms that matched
            final_priority = round(
                (priority_percentage * 1.5)
                + illness_confidence
                - (0.05 * (10 - len(matched_symptoms))),
                2,
            )

            # Boost priority if all symptoms match
            if len(matched_symptoms) == len(rule["symptoms"]):
                final_priority += 0.5  # Small priority boost for perfect matches
            # Incorporate Pet Characteristics into the Ranking with STRONGER Boost
            # Apply a stronger penalty for illnesses with strict age restrictions that don’t match
            if rule["age_range"] == pet_info["age_range"]:
                age_match = 1.3  # Strongest boost for exact age match
            elif rule["age_range"] == "Any":
                age_match = 1.15  # Can apply to any pet, slight boost
            elif rule["age_range"] in ["Puppy", "Adult", "Senior"]:  
                age_match = 0.5  # HARSH penalty for incorrect age
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

            # Calculate an overall pet characteristic match factor with higher impact
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
    # Reduce confidence for illnesses with high symptom overlap
    for illness in possible_diagnoses:
        for competitor in possible_diagnoses:
            if illness != competitor:
                shared_symptoms = set(illness["matched_symptoms"]) & set(
                    competitor["matched_symptoms"]
                )
                if (
                    competitor["confidence"] < illness["confidence"]
                ):  # Only penalize lower-ranked illnesses
                    competitor["confidence"] *= 0.85
                if len(shared_symptoms) >= 3:
                    competitor[
                        "confidence"
                    ] *= 0.85  # Reduce confidence for competing illnesses
                elif len(shared_symptoms) >= 5:
                    competitor[
                        "confidence"
                    ] *= 0.75  # If overlap is too high, reduce further

    # Stronger emphasis on priority in ranking
    possible_diagnoses.sort(
        key=lambda x: -((x["confidence"] * 2.5) + (x["priority"] * 1.3))
    )

    # Store results in fact base
    fact_base["possible_diagnosis"] = possible_diagnoses
    save_fact_base(fact_base)

    # Print results
    print("\n🩺 **Diagnosis Results:**")
    for diagnosis in possible_diagnoses:
        print(
            f"🔹 {diagnosis['illness']} (Confidence: {diagnosis['confidence']}, Priority: {diagnosis['priority']})"
        )

    print("\n✅ Diagnosis completed and saved to fact_base.json")


# Run the algorithm
if __name__ == "__main__":
    forward_chaining()
