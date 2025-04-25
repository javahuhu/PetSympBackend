import json

# File paths
input_file = "new list of illnesses.txt"
output_file = "unique_symptoms.json"

unique_symptoms = set()

with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        # Skip empty lines and illness names (those without commas)
        if "," not in line or not line:
            continue
        # Split and clean symptom names
        symptoms = [sym.strip() for sym in line.split(",")]
        unique_symptoms.update(symptoms)

# Sort alphabetically
sorted_symptoms = sorted(unique_symptoms)

# Save to JSON
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(sorted_symptoms, json_file, indent=2, ensure_ascii=False)

print(
    f"✅ Successfully extracted {len(sorted_symptoms)} unique symptoms into '{output_file}'"
)
