import json
import csv
import os

# Define the relative folder where the data files are stored
data_folder = "data"

# Construct file paths using relative paths
json_filepath = os.path.join(data_folder, "new_knowledge_base.json")
csv_filepath = os.path.join(data_folder, "knowledge_base.csv")

# Load JSON file
with open(json_filepath, "r", encoding="utf-8") as file:
    data = json.load(file)

# Ensure "rules" exists and is a list
if "rules" not in data or not isinstance(data["rules"], list):
    raise ValueError("🚨 ERROR: 'rules' key missing or not a list in JSON!")

# Extract illness list
illnesses = data["rules"]

print(f"✅ JSON successfully loaded. Total illnesses: {len(illnesses)}")

# Define CSV headers
headers = ["Illness", "Symptom", "Stage", "Weight", "Severity", "Priority", "Age Range", "Breed", "Size", "Confidence"]

# Ensure the "data" folder exists before writing the CSV
os.makedirs(data_folder, exist_ok=True)

# Create CSV file
with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(headers)

    # Iterate through each illness entry
    for illness_entry in illnesses:
        illness_name = illness_entry.get("illness", "Unknown")
        age_range = illness_entry.get("age_range", "Any")
        breed = illness_entry.get("breed", "Any")
        size = illness_entry.get("size", "Any")
        confidence = illness_entry.get("confidence", "N/A")

        # Iterate through symptoms
        for symptom in illness_entry.get("symptoms", []):
            symptom_name = symptom.get("name", "Unknown Symptom")
            stage = symptom.get("stage", "Any Stage")
            weight = symptom.get("weight", 0)
            severity = symptom.get("severity", "Unknown")
            priority = symptom.get("priority", 0)

            # Write row for each symptom
            csv_writer.writerow([illness_name, symptom_name, stage, weight, severity, priority, age_range, breed, size, confidence])

print(f"✅ CSV file created successfully: {csv_filepath}")
