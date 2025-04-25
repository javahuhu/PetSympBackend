import pandas as pd
import random

# Master list of symptoms
symptom_list = [
    "Abdominal Bloating", "Abdominal Pain", "Aggression", "Anasarca", "Anorexia", "Anxiety", "Apathy", "Ataxia",
    "Biting at Exposure Site", "Blindness", "Bloody Diarrhea", "Breathing Difficulty", "Bruised Nose and Mouth",
    "Circling", "Coma", "Confusion", "Convulsions", "Coughing", "Decreased Appetite", "Decreased Suckling",
    "Dehydration", "Depression", "Diarrhea", "Disorientation", "Drooping Eyelid", "Dysphagia", "Encephalitis",
    "Enlarged Lymph Nodes", "Enlarged Spleen", "Enlarged Tonsils", "Excessive Crying", "Excessive Drooling",
    "Excessive Salivation", "Eye Discharge", "Eye Inflammation", "Facial Itching", "Facial Paralysis", "Fearfulness",
    "Fever", "Gagging", "Genital Lesions", "Gum Hemorrhages", "Hallucinations", "Hardened Footpads", "Head Pressing",
    "Head Tilt", "Hemorrhagic Diarrhea", "Hyperesthesia", "Hydrophobia", "Hypersalivation", "Hypothermia",
    "Intense Itching", "Irritability", "Itching", "Jaundice", "Lethargy", "Light Sensitivity", "Listlessness",
    "Loss of Appetite", "Low White Blood Cell Count", "Miscarriage", "Mucus in Stool", "Muscle Paralysis",
    "Muscle Spasms", "Muscle Twitching", "Myocarditis", "Nasal Discharge", "Orange-Tinted Stool", "Paralysis",
    "Petechiae on Gums", "Petechiae on Skin", "Photophobia", "Pneumonia", "Pruritus", "Profuse Vomiting",
    "Purulent Nasal Discharge", "Rashes", "Red Spots on Skin", "Restlessness", "Retching", "Runny Eyes",
    "Runny Nose", "Seizures", "Severe Depression", "Severe Diarrhea", "Severe Vomiting", "Severe White Blood Cell Reduction",
    "Sneezing", "Sound Sensitivity", "Spontaneous Bleeding", "Staggering", "Stillbirths", "Sudden Diarrhea",
    "Swallowing Difficulty", "Swollen Head", "Swollen Neck", "Swollen Trunk", "Thirst", "Tremors",
    "Uncharacteristic Aggression", "Uncharacteristic Friendliness", "Vocalization", "Vomiting", "Wandering Aimlessly",
    "Watery Eye Discharge", "Watery Nasal Discharge", "Weakness", "Weight Loss", "Yellow Ears", "Yellow Gums", "Yellow Skin"
]

# Probability tiers for symptoms
tier_probabilities = {
    "Tier 0": 1.00,  # Always present
    "Tier 1": random.uniform(0.70, 0.99),
    "Tier 2": random.uniform(0.35, 0.69),
    "Tier 3": random.uniform(0.11, 0.34),
    "Tier 4": random.uniform(0.01, 0.10),
    "Tier 5": 0.00  # Always absent
}

# Complete dictionary of illnesses with their symptom probability tiers
illnesses = {
    "Canine Coronavirus": {
        "Tier 0": ["Anorexia", "Diarrhea"],
        "Tier 1": ["Bloody Diarrhea", "Dehydration", "Lethargy", "Loss of Appetite", "Sudden Diarrhea", "Vomiting"],
        "Tier 2": ["Depression", "Foul-Smelling Stool"],
        "Tier 3": ["Mucus in Stool", "Orange-Tinted Stool"],
        "Tier 4": []
    },
    "Canine Distemper": {
        "Tier 0": ["Fever"],
        "Tier 1": ["Coughing", "Nasal Discharge", "Eye Discharge", "Lethargy", "Anorexia", "Vomiting", "Diarrhea"],
        "Tier 2": ["Muscle Twitching", "Seizures", "Paralysis", "Hardened Footpads", "Depression"],
        "Tier 3": ["Pneumonia", "Weight Loss"],
        "Tier 4": []
    },
    "Canine Herpesvirus": {
        "Tier 0": ["Nasal Discharge"],
        "Tier 1": ["Abdominal Pain", "Breathing Difficulty", "Coughing", "Decreased Appetite", "Diarrhea", "Eye Discharge", "Fever", "Lethargy", "Weakness", "Weight Loss"],
        "Tier 2": ["Abdominal Bloating", "Decreased Suckling", "Listlessness", "Pneumonia"],
        "Tier 3": ["Excessive Crying", "Eye Inflammation", "Genital Lesions", "Gum Hemorrhages", "Petechiae on Gums", "Rashes", "Seizures", "Sneezing"],
        "Tier 4": ["Miscarriage", "Stillbirths"]
    },
    "Canine Influenza": {
        "Tier 0": ["Coughing"],
        "Tier 1": ["Nasal Discharge", "Fever", "Lethargy", "Decreased Appetite", "Breathing Difficulty"],
        "Tier 2": ["Purulent Nasal Discharge", "Eye Discharge", "Sneezing"],
        "Tier 3": [],
        "Tier 4": []
    },
    "Kennel Cough": {
        "Tier 0": ["Coughing"],
        "Tier 1": ["Decreased Appetite", "Eye Discharge", "Fever", "Lethargy", "Nasal Discharge", "Sneezing"],
        "Tier 2": ["Gagging", "Retching", "Runny Nose"],
        "Tier 3": [],
        "Tier 4": []
    },
    "Canine Parvovirus": {
        "Tier 0": ["Bloody Diarrhea", "Severe Vomiting"],
        "Tier 1": ["Lethargy", "Loss of Appetite", "Fever"],
        "Tier 2": ["Dehydration"],
        "Tier 3": ["Abdominal Pain"],
        "Tier 4": []
    },
    "Rabies": {
        "Tier 0": ["Aggression"],
        "Tier 1": ["Anxiety", "Biting at Exposure Site", "Confusion", "Decreased Appetite", "Breathing Difficulty", "Swallowing Difficulty", "Excessive Drooling", "Excessive Salivation", "Fearfulness", "Fever", "Hallucinations", "Hydrophobia", "Irritability", "Lethargy", "Muscle Paralysis", "Muscle Twitching", "Restlessness", "Seizures", "Staggering", "Uncharacteristic Aggression", "Uncharacteristic Friendliness", "Vomiting", "Weakness"],
        "Tier 2": [],
        "Tier 3": ["Light Sensitivity", "Sound Sensitivity"],
        "Tier 4": []
    },
    "Canine Minute Virus": {
        "Tier 0": ["Bloody Diarrhea"],
        "Tier 1": ["Abdominal Pain", "Anasarca", "Anorexia", "Dehydration", "Breathing Difficulty", "Fever", "Hypothermia", "Lethargy", "Loss of Appetite", "Severe Diarrhea", "Weight Loss"],
        "Tier 2": ["Profuse Vomiting"],
        "Tier 3": [],
        "Tier 4": []
    },
    "Infectious Canine Hepatitis": {
        "Tier 0": ["Vomiting"],
        "Tier 1": ["Abdominal Pain", "Anorexia", "Apathy", "Ataxia", "Blindness", "Bruised Nose and Mouth", "Coughing", "Dehydration", "Depression", "Diarrhea", "Eye Discharge", "Nasal Discharge", "Fever", "Hemorrhagic Diarrhea", "Jaundice", "Lethargy", "Low White Blood Cell Count", "Severe Depression"],
        "Tier 2": ["Enlarged Tonsils", "Enlarged Lymph Nodes", "Enlarged Spleen", "Petechiae on Skin", "Red Spots on Skin", "Seizures", "Severe White Blood Cell Reduction", "Spontaneous Bleeding", "Thirst", "Watery Eye Discharge", "Watery Nasal Discharge", "Yellow Ears", "Yellow Gums", "Yellow Skin"],
        "Tier 3": ["Swollen Head", "Swollen Neck", "Swollen Trunk"],
        "Tier 4": []
    },
    "Pseudorabies": {
        "Tier 0": [],
        "Tier 1": ["Aggression", "Ataxia", "Circling", "Coma", "Convulsions", "Depression", "Breathing Difficulty", "Disorientation", "Swallowing Difficulty", "Excessive Salivation", "Facial Paralysis", "Facial Itching", "Fever", "Head Pressing", "Head Tilt", "Hyperesthesia", "Hypersalivation", "Intense Itching", "Irritability", "Lethargy", "Muscle Spasms", "Paralysis", "Light Sensitivity", "Itching", "Drooping Eyelid", "Restlessness", "Seizures", "Tremors", "Vocalization", "Vomiting", "Wandering Aimlessly", "Weakness"],
        "Tier 2": [],
        "Tier 3": [],
        "Tier 4": []
    }
}

# Generate dataset
data = []

for illness, tiers in illnesses.items():
    for case_num in range(1, 51):  # 50 test cases per illness
        case = {"Test Case ID": f"{illness}_{case_num}", "Illness": illness}
        
        for symptom in symptom_list:
            assigned = 0  # Default: symptom is absent
            
            for tier, symptoms in tiers.items():
                if symptom in symptoms:
                    probability = tier_probabilities[tier]
                    assigned = 1 if random.random() < probability else 0
                    break
            
            case[symptom] = assigned
        
        data.append(case)

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv("canine_illness_dataset.csv", index=False)

print("Dataset created successfully: 'canine_illness_dataset.csv'")
