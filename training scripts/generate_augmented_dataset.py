import pandas as pd
import joblib
import json
from tqdm import tqdm

# Load dataset
df = pd.read_csv("latest.csv")
original_df = df.copy()

# Load knowledge base and GB model
with open("data/updated_knowledge_base_v2_fixed.json", "r") as f:
    knowledge_base = json.load(f)["rules"]

boosting_model = joblib.load("gradient_model.pkl")

# Extract symptom columns
symptom_cols = [col for col in df.columns if col != "Illness"]

# Prep feature containers
fc_scores = []
gb_scores = []
symptom_match_ratios = []


# Simulate FC scoring
def simulate_fc(symptom_row, illness_name):
    rule = next((r for r in knowledge_base if r["illness"] == illness_name), None)
    if not rule:
        return 0.0, 0.0

    total_weight = sum(s["weight"] for s in rule["symptoms"])
    rule_symptom_names = [s["name"] for s in rule["symptoms"]]

    matched = [s for s in rule["symptoms"] if symptom_row.get(s["name"], 0) == 1]
    matched_weight = sum(s["weight"] for s in matched)

    # ✅ Corrected: ratio is matched rule symptoms / total rule symptoms
    ratio = len(matched) / len(rule_symptom_names) if rule_symptom_names else 0.0
    fc_conf = (
        round((matched_weight / total_weight) * rule["confidence"], 2)
        if total_weight > 0
        else 0.0
    )

    return fc_conf, ratio


# Generate scores per row
for _, row in tqdm(df.iterrows(), total=len(df)):
    illness = row["Illness"]
    symptom_vector = row[symptom_cols].to_dict()

    # FC + Ratio
    fc_conf, match_ratio = simulate_fc(symptom_vector, illness)
    fc_scores.append(fc_conf)
    symptom_match_ratios.append(match_ratio)

    # GB
    input_df = pd.DataFrame([symptom_vector])
    input_df = input_df.reindex(columns=boosting_model.feature_names_in_, fill_value=0)
    probas = boosting_model.predict_proba(input_df)[0]
    illness_index = list(boosting_model.classes_).index(illness)
    gb_scores.append(round(probas[illness_index], 4))

# Append new columns to dataset
original_df["FC_Confidence"] = fc_scores
original_df["GB_Confidence"] = gb_scores
original_df["Symptom_Match_Ratio"] = symptom_match_ratios

# Save to CSV
original_df.to_csv("latest_augmented.csv", index=False)
print("\n✅ Saved: latest_augmented.csv with FC, GB, and Symptom Ratio features.")
