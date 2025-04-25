import pandas as pd
import random

# Define probability tiers for symptom occurrence
SYMPTOM_PROBABILITIES = {
    "core": (0.8, 1.0),  # Core symptoms appear in 80-100% of cases
    "common": (0.5, 0.8),  # Common symptoms appear in 50-80% of cases
    "rare": (0.1, 0.5),  # Rare symptoms appear in 10-50% of cases
}

# Define all unique symptoms across viral diseases
all_symptoms = sorted(
    [
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
)

# Define symptom-specific stages for each illness
symptom_stage_mapping = {
    "Canine Coronavirus": {
        "Diarrhea": "Any Stage",
        "Vomiting": "Any Stage",
        "Lethargy": "Any Stage",
        "Bloody Stool": "Any Stage",
        "Loss of Appetite": "Any Stage",
        "Fever": "Early",
        "Abdominal Pain": "Any Stage",
        "Dehydration": "Late",
        "Seizures": "Late",
    },
    "Canine Distemper": {
        "Coughing": "Early",
        "Nasal Discharge": "Early",
        "Lethargy": "Early",
        "Eye Discharge": "Early",
        "Diarrhea": "Any Stage",
        "Vomiting": "Any Stage",
        "Sneezing": "Early",
        "Seizures": "Late",
        "Muscle Spasms": "Late",
        "Depression": "Any Stage",
        "Weight Loss": "Late",
        "Paralysis": "Late",
        "Thick Substance in Eyes": "Late",
    },
    "Canine Herpesvirus": {
        "Nasal Discharge": "Early",
        "Vomiting": "Any Stage",
        "Diarrhea": "Any Stage",
        "Weakness": "Any Stage",
        "Loss of Appetite": "Any Stage",
        "Difficulty Breathing": "Late",
    },
    "Canine Influenza": {
        "Coughing": "Early",
        "Nasal Discharge": "Early",
        "Fever": "Early",
        "Sneezing": "Early",
        "Lethargy": "Any Stage",
        "Loss of Appetite": "Any Stage",
        "Runny Nose": "Any Stage",
        "Eye Discharge": "Late",
    },
    "Kennel Cough": {
        "Coughing": "Early",
        "Sneezing": "Early",
        "Nasal Discharge": "Early",
        "Gagging": "Any Stage",
        "Forceful Cough": "Any Stage",
        "Loss of Appetite": "Any Stage",
        "Lethargy": "Any Stage",
        "Chest Pain": "Late",
    },
    "Canine Parvovirus": {
        "Diarrhea": "Any Stage",
        "Vomiting": "Any Stage",
        "Lethargy": "Early",
        "Bloody Stool": "Any Stage",
        "Loss of Appetite": "Any Stage",
        "Fever": "Early",
        "Abdominal Pain": "Any Stage",
        "Dehydration": "Late",
        "Weakness": "Any Stage",
    },
    "Rabies": {
        "Aggression": "Any Stage",
        "Paralysis": "Late",
        "Seizures": "Late",
        "Drooling": "Any Stage",
        "Fever": "Early",
        "Lethargy": "Any Stage",
        "Loss of Appetite": "Any Stage",
        "Depression": "Any Stage",
        "Sensitivity to Light": "Late",
        "Hydrophobia": "Late",
        "Difficulty Breathing": "Late",
        "Difficulty Swallowing": "Late",
        "Lack of Coordination": "Late",
    },
    "Canine Minute Virus": {
        "Diarrhea": "Any Stage",
        "Vomiting": "Any Stage",
        "Lethargy": "Any Stage",
        "Weight Loss": "Late",
        "Loss of Appetite": "Any Stage",
    },
    "Infectious Canine Hepatitis": {
        "Fever": "Early",
        "Vomiting": "Any Stage",
        "Lethargy": "Early",
        "Loss of Appetite": "Any Stage",
        "Abdominal Pain": "Any Stage",
        "Coughing": "Early",
        "Jaundice": "Late",
        "Pale Gums": "Late",
        "Tender Abdomen": "Late",
        "Corneal Edema": "Late",
        "Bleeding Disorders": "Late",
    },
    "Pseudorabies": {
        "Itching": "Early",
        "Fever": "Early",
        "Vomiting": "Any Stage",
        "Seizures": "Late",
        "Paralysis": "Late",
    },
}


# Function to generate multiple instances per illness with symptom-specific stages
def generate_variable_illness_cases_with_stages(
    illness_name, symptoms, symptom_stages, num_cases=50
):
    cases = []
    for _ in range(num_cases):
        case = {symptom: 0 for symptom in all_symptoms}  # Initialize all symptoms as 0
        case_stages = {
            f"{symptom}_Stage": "None" for symptom in all_symptoms
        }  # Initialize stages as "None"

        # Assign symptoms based on probability tiers
        for category, symptom_list in {
            "core": symptoms[:3],
            "common": symptoms[3:6],
            "rare": symptoms[6:],
        }.items():
            prob_range = SYMPTOM_PROBABILITIES[category]
            for symptom in symptom_list:
                if random.uniform(0, 1) < random.uniform(*prob_range):
                    case[symptom] = 1  # Symptom is present
                    case_stages[f"{symptom}_Stage"] = symptom_stages[
                        symptom
                    ]  # Assign correct stage

        # Assign severity, priority, and confidence with variations
        case["Severity"] = random.choice(["Low", "Medium", "High"])
        case["Priority"] = round(random.uniform(1.3, 1.8), 1)
        case["Illness"] = illness_name
        case["Confidence"] = round(random.uniform(0.75, 0.95), 2)

        # Merge symptoms and their stages
        case.update(case_stages)
        cases.append(case)

    return cases


# Generate dataset with symptom variability and correct stage mapping
variable_viral_diseases_dataset = []
for illness, symptoms in symptom_stage_mapping.items():
    variable_viral_diseases_dataset.extend(
        generate_variable_illness_cases_with_stages(
            illness, list(symptoms.keys()), symptoms, 50
        )
    )

# Convert to DataFrame
df_variable_viral_diseases = pd.DataFrame(variable_viral_diseases_dataset)

# Save as CSV file
df_variable_viral_diseases.to_csv("updated_viral_diseases_dataset.csv", index=False)

print(
    "Dataset successfully generated and saved as 'updated_viral_diseases_dataset.csv'"
)
