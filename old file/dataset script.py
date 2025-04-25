import json
import pandas as pd

# Load the knowledge base JSON file
with open(
    "data\knowledge_base.json", "r"
) as file:  # Ensure the JSON file is in the same directory
    knowledge_base = json.load(file)

# Extract illnesses and their corresponding symptoms
dataset = []
all_symptoms = set()

# Collect all possible symptoms
for rule in knowledge_base["rules"]:
    for symptom in rule["symptoms"]:
        all_symptoms.add(symptom["name"])

# Convert to list for column ordering
all_symptoms = sorted(list(all_symptoms))

# Process each illness and extract structured data with updated Stage encoding
updated_dataset = []

for rule in knowledge_base["rules"]:
    illness_name = rule["illness"]
    illness_confidence = rule["confidence"]

    # Create a default row for all symptoms set to 0 (not present)
    illness_data = {symptom: 0 for symptom in all_symptoms}

    # Add new binary fields for Early & Late stages
    illness_data["Early_Stage"] = 0
    illness_data["Late_Stage"] = 0

    # Populate row with symptom data
    for symptom in rule["symptoms"]:
        symptom_name = symptom["name"]
        illness_data[symptom_name] = 1  # Mark symptom presence
        illness_data["Severity"] = symptom["severity"]  # Record severity level
        illness_data["Priority"] = symptom["priority"]  # Record priority

        # Update Early and Late stage encoding
        if symptom["stage"] == "Early":
            illness_data["Early_Stage"] = 1
        elif symptom["stage"] == "Late":
            illness_data["Late_Stage"] = 1
        elif symptom["stage"] == "Any Stage":
            illness_data["Early_Stage"] = 1
            illness_data["Late_Stage"] = 1  # Both are set to 1 for Any Stage

    # Add illness label and confidence
    illness_data["Illness"] = illness_name
    illness_data["Confidence"] = illness_confidence

    updated_dataset.append(illness_data)

# Convert to DataFrame
df_updated = pd.DataFrame(updated_dataset)

# Convert categorical values into numerical codes
severity_mapping = {"Low": 1, "Medium": 2, "High": 3}
df_updated["Severity"] = df_updated["Severity"].map(severity_mapping)

# Fill any missing values with default 0
df_updated = df_updated.fillna(0)

# Save as CSV file
df_updated.to_csv("extracted_dataset.csv", index=False)

print("Dataset successfully extracted and saved as 'extracted_dataset.csv'")
