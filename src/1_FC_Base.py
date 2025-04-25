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
    print("\n🔹 Enter your pet's symptoms (comma-separated, e.g., Vomiting, Loss of Appetite):")
    user_symptoms = input("➡ Symptoms: ").strip().split(",")
    user_symptoms = [symptom.strip().lower() for symptom in user_symptoms]

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

# Basic Forward Chaining Algorithm without Boosting or Weight Adjustments
def forward_chaining():
    knowledge_base = load_knowledge_base()
    fact_base = get_user_input()
    user_symptoms = fact_base["symptoms"]
    
    possible_diagnoses = []
    
    for rule in knowledge_base:
        matched_symptoms = [symptom for symptom in rule["symptoms"] if symptom["name"].lower() in user_symptoms]
        
        if matched_symptoms:
            symptom_match_ratio = len(matched_symptoms) / len(rule["symptoms"])
            final_confidence = round(rule["confidence"] * symptom_match_ratio, 2)
            
            possible_diagnoses.append({
                "illness": rule["illness"],
                "matched_symptoms": [s["name"] for s in matched_symptoms],
                "confidence": final_confidence,
                "priority": rule.get("priority", 0)  # Default priority to 0 if missing
            })
    
    # Sort results by confidence only (no multipliers or extra factors)
    possible_diagnoses.sort(key=lambda x: -x["confidence"])
    
    fact_base["possible_diagnosis"] = possible_diagnoses
    save_fact_base(fact_base)
    
    # Print results
    print("\n🩺 **Diagnosis Results:**")
    for diagnosis in possible_diagnoses:
        print(f"🔹 {diagnosis['illness']} (Confidence: {diagnosis['confidence']})")
    
    print("\n✅ Diagnosis completed and saved to fact_base.json")

# Run the algorithm
if __name__ == "__main__":
    forward_chaining()
