import os
import re
import json
import smtplib
import uvicorn
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import firebase_admin
from firebase_admin import auth, credentials

# ───── Initialization ─────

if not firebase_admin._apps:
    cred = credentials.Certificate("petsympsdk.json")
    firebase_admin.initialize_app(cred)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───── File Paths & Static Loads ─────

FACT_BASE_PATH             = "data/fact_base.json"
KNOWLEDGE_BASE_PATH        = "data/updated_knowledge_base_v2_fixed.json"
ILLNESS_INFO_PATH          = "data/expanded_illness_info_complete.json"
FOLLOWUP_QUESTIONS_PATH    = "data/updated_follow_up_questions_tuned.json"
BREED_CATEGORY_MAP_PATH    = "data/breed_category_mapping.json"
DATASET_PATH               = "data/latest_augmented.csv"

def load_json(p):
    with open(p) as f:
        return json.load(f)

kb_rules               = load_json(KNOWLEDGE_BASE_PATH)["rules"]
illness_info_db        = load_json(ILLNESS_INFO_PATH)
symptom_followups      = load_json(FOLLOWUP_QUESTIONS_PATH)
breed_category_mapping = load_json(BREED_CATEGORY_MAP_PATH)

boosting_model    = joblib.load("model/gradient_model.pkl")
adaboost_model    = joblib.load("model/adaboost_model.pkl")
selected_features = joblib.load("model/adaboost_selected_features.pkl")

df = pd.read_csv(DATASET_PATH)
all_symptoms = [c for c in df.columns if c != "Illness"]

# ───── OTP / Password Reset Endpoints ─────

OTP_STORE    = {}
EMAIL        = "petsymp0@gmail.com"
APP_PASSWORD = "gqox rtam taom hhbb"

class OTPRequest(BaseModel):
    email: str
    otp:   str

@app.post("/send-otp")
async def send_otp(req: OTPRequest):
    try:
        OTP_STORE[req.email] = req.otp
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your PetSymp OTP Code"
        msg["From"]    = EMAIL
        msg["To"]      = req.email
        html = f"""
        <html><body>
          <h2>PetSymp Email Verification</h2>
          <p>Your OTP is <strong style='color:#52AAA4'>{req.otp}</strong></p>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL, APP_PASSWORD)
            server.sendmail(EMAIL, req.email, msg.as_string())
        return {"message": "OTP sent successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send OTP: {e}")

@app.post("/reset-password")
async def reset_password(r: Request):
    data = await r.json()
    email, newpw, otp = data.get("email"), data.get("newPassword"), data.get("otp")
    if not all([email, newpw, otp]):
        raise HTTPException(status_code=400, detail="Email, new password, and OTP are required.")
    if OTP_STORE.get(email) != otp:
        raise HTTPException(status_code=401, detail="Invalid OTP.")
    try:
        user = auth.get_user_by_email(email)
        auth.update_user(user.uid, password=newpw)
        del OTP_STORE[email]
        return {"message": "Password updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update password: {e}")

# ───── Utility Functions ─────

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

def validate_size(sz):
    return sz.capitalize() if isinstance(sz, str) and sz.lower() in ["small","medium","large"] else "Medium"

def parse_duration_range(range_str):
    """
    Parses a duration range string like "1-4 days" into a tuple of (1, 4).
    If parsing fails, returns None.
    """
    if not range_str:
        return None
    cleaned = range_str.lower().replace("days", "").replace("day", "").strip()
    parts = cleaned.split("-")
    if len(parts) == 2:
        try:
            lower = float(parts[0].strip())
            upper = float(parts[1].strip())
            return (lower, upper)
        except ValueError:
            return None
    return None

def duration_overlap(user_range_str, expected_range_str):
    """
    Returns True if the numeric intervals from user_range_str and expected_range_str overlap.
    For example, "1-4 days" and "2-4 days" would overlap.
    """
    user_interval = parse_duration_range(user_range_str)
    expected_interval = parse_duration_range(expected_range_str)
    if user_interval and expected_interval:
        return max(user_interval[0], expected_interval[0]) <= min(
            user_interval[1], expected_interval[1]
        )
    return False

def align_features(feature_vector, expected_features):
    """Align feature vector with expected features to ensure consistency."""
    feature_vector = feature_vector.copy()  # Prevents SettingWithCopyWarning

    # Add missing features as zeros
    for feature in expected_features:
        if feature not in feature_vector.columns:
            feature_vector.loc[:, feature] = 0

    # Reorder columns to match training feature order
    return feature_vector[expected_features]

def compute_subtype_coverage(rule, user_answers):
    """
    Computes a normalized subtype coverage score for an illness.
    For each symptom that has subtype data in the knowledge base, if the user's answer
    (from any key in user_answers for that symptom) is found in the expected set,
    count it as a full match (1), otherwise 0.
    The coverage score is returned as a percentage.
    """
    matched = 0
    total = 0
    for symptom in rule.get("symptoms", []):
        symptom_name = symptom["name"].lower().strip()
        if "subtype" in symptom and symptom["subtype"]:
            total += 1
            expected = {
                sub.strip().lower()
                for sub in symptom["subtype"].split(",")
                if sub.strip()
            }
            user_response = user_answers.get(symptom_name, {})
            user_subtype = None
            for key, val in user_response.items():
                candidate = val.lower().strip()
                if candidate in expected:
                    user_subtype = candidate
                    break
            if user_subtype:
                matched += 1
    if total == 0:
        return 0
    return (matched / total) * 100

def adjust_confidence_with_followups(confidence, symptom_details, illness_name, user_answers):
    """Modify confidence scores based on user follow-up answers and KB expectations."""
    illness_rule = next((r for r in kb_rules if r["illness"] == illness_name), None)
    if not illness_rule:
        return confidence

    total_multiplier = 1.0

    for symptom in symptom_details:
        symptom_name = (symptom["name"] if isinstance(symptom, dict) else symptom).lower().strip()

        if symptom_name in symptom_followups and illness_rule:
            expected_symptoms = {s["name"].lower(): s for s in illness_rule["symptoms"]}
            if symptom_name in expected_symptoms:
                expected_data = expected_symptoms[symptom_name]
                expected_duration = expected_data.get("duration_range", "Any")
                expected_severity = expected_data.get("severity", "Any")
                expected_subtypes = [
                    sub.strip().lower()
                    for sub in expected_data.get("subtype", "Any").split(",")
                    if sub.strip()
                ]

                user_response = user_answers.get(symptom_name, {})
                user_duration = user_response.get(
                    f"How long has your pet had {symptom_name}?", None
                )
                user_severity = user_response.get(
                    f"Is the {symptom_name} Mild, Moderate, or Severe?", None
                )

                user_subtype = None
                if symptom_name in symptom_followups:
                    for question in symptom_followups[symptom_name]["questions"]:
                        answer = user_response.get(question, None)
                        if answer:
                            candidate = answer.lower().strip()
                            if expected_subtypes and candidate in expected_subtypes:
                                user_subtype = candidate
                                break
                if not user_subtype and user_severity:
                    candidate = user_severity.lower().strip()
                    if expected_subtypes and candidate in expected_subtypes:
                        user_subtype = candidate
                user_subtype_clean = user_subtype if user_subtype else None

                impact_values = symptom_followups.get(symptom_name, {}).get(
                    "impact", {}
                )
                severity_impact = (
                    impact_values.get(user_severity.lower(), 1.2)
                    if user_severity
                    else 1.2
                )
                subtype_impact = (
                    impact_values.get(user_subtype_clean, 1.2)
                    if user_subtype_clean
                    else 1.2
                )

                if user_duration and expected_duration.lower() != "any":
                    if duration_overlap(user_duration, expected_duration):
                        duration_impact = impact_values.get(user_duration.lower(), 1.2)
                    else:
                        duration_impact = 0.95
                else:
                    duration_impact = 1.2

                kb_match_bonus = 1.0
                if user_severity and expected_severity.lower() != "any":
                    kb_match_bonus *= (
                        1.02
                        if user_severity.lower() == expected_severity.lower()
                        else 0.95
                    )

                if user_duration and expected_duration.lower() != "any":
                    if duration_overlap(user_duration, expected_duration):
                        kb_match_bonus *= 1.03
                    else:
                        kb_match_bonus *= 0.95

                if user_subtype_clean and expected_subtypes != ["any"]:
                    kb_match_bonus *= (
                        1.08 if user_subtype_clean in expected_subtypes else 0.9
                    )

                total_multiplier *= (
                    severity_impact * subtype_impact * duration_impact * kb_match_bonus
                )

    return round(confidence * total_multiplier, 2)

def normalize_fact_base(fb):
    # Get or create pet_info dictionary
    pi = fb.get("pet_info", {})
    
    # Handle age - use age_years as the source of truth
    age = pi.get("age_years") or pi.get("age")
    if age is not None:
        # Store only age_years, not both age and age_years
        pi["age_years"] = str(age)
        # Remove age if it exists to avoid duplication
        if "age" in pi:
            del pi["age"]
    
    # Set age_range based on age_years
    pi["age_range"] = categorize_age(pi.get("age_years", "0"))
    
    # Ensure size is properly formatted
    pi["size"] = validate_size(pi.get("size", "Medium"))
    
    # Ensure breed is properly capitalized
    pi["breed"] = pi.get("breed", "Any").strip().capitalize()
    
    # Update pet_info in fact_base
    fb["pet_info"] = pi

    # Normalize symptoms
    syms = fb.get("symptoms", [])
    norm = []
    for s in syms:
        if isinstance(s, str):
            norm.append({"name": s.lower().strip()})
        elif isinstance(s, dict) and "name" in s:
            norm.append({"name": s["name"].lower().strip()})
    fb["symptoms"] = norm

    # Normalize user_answers
    ua = fb.get("user_answers", {})
    nua = {}
    for sym, ans in ua.items():
        key = sym.lower().strip()
        if isinstance(ans, dict):
            nua[key] = {q.strip(): a for q, a in ans.items()}
    fb["user_answers"] = nua
    
    return fb

# ───── Pipeline & Diagnose ─────

def forward_chaining(fact_base):
    """Pure symptom-based rule matching using Forward Chaining."""
    user_symptoms = [
        s["name"].lower().strip()
        for s in fact_base["symptoms"]
        if isinstance(s, dict) and "name" in s
    ]

    possible_diagnoses = []

    for rule in kb_rules:
        total_weight = sum(symptom["weight"] for symptom in rule["symptoms"])
        matched_weight = sum(
            symptom["weight"]
            for symptom in rule["symptoms"]
            if symptom["name"].lower().strip() in user_symptoms
        )

        if matched_weight > 0:
            match_ratio = matched_weight / total_weight
            confidence_fc = round(rule["confidence"] * match_ratio, 2)

            possible_diagnoses.append(
                {
                    "illness": rule["illness"],
                    "matched_symptoms": [
                        s["name"].lower().strip()
                        for s in rule["symptoms"]
                        if s["name"].lower().strip() in user_symptoms
                    ],
                    "confidence_fc": confidence_fc,
                }
            )

    return possible_diagnoses

def gradient_boosting_ranking(possible_diagnoses, fact_base):
    """Uses Gradient Boosting to refine illness ranking based on symptom match and model prediction."""
    user_symptoms = (
        [
            s["name"].lower().strip()
            for s in fact_base["symptoms"]
            if isinstance(s, dict) and "name" in s
        ]
        if isinstance(fact_base["symptoms"], list)
        else []
    )

    pet_info = fact_base["pet_info"]
    user_answers = fact_base["user_answers"]

    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # Create feature vector for ML model (using only symptoms)
        feature_vector = pd.DataFrame(
            [
                [
                    1 if symptom.lower() in user_symptoms else 0
                    for symptom in all_symptoms
                ]
            ],
            columns=all_symptoms,
        )

        feature_vector_selected = align_features(
        feature_vector,
        boosting_model.feature_names_in_
)


        # Get Gradient Boosting prediction
        raw_predictions = boosting_model.predict_proba(feature_vector_selected)
        illness_index = list(boosting_model.classes_).index(diagnosis["illness"])
        gb_confidence = raw_predictions[0][illness_index]

        # Retrieve illness rule safely
        rule = next(
            (r for r in kb_rules if r["illness"] == diagnosis["illness"]),
            None,
        )

        # Default multipliers for age, breed, and size
        if rule:
            age_match = (
                1.1
                if rule.get("age_range", "Any").lower()
                == pet_info.get("age_range", "Any").lower()
                else 0.85
            )

            rule_breed = rule.get("breed", "Any").strip().lower()
            pet_breed = pet_info.get("breed", "Any").strip().lower()
            pet_breed_category = breed_category_mapping.get(
                pet_breed.title(), ""
            ).lower()

            if rule_breed == "any":
                breed_match = 1.0
            elif rule_breed == pet_breed:
                breed_match = 1.1
            elif rule_breed == pet_breed_category:
                breed_match = 1.1
            else:
                breed_match = 0.9

            size_match = (
                1.1
                if rule.get("size", "Any").lower()
                == pet_info.get("size", "Any").lower()
                else 0.9
            )
        else:
            age_match = breed_match = size_match = 1.0

        final_confidence_gb = adjust_confidence_with_followups(
            round(gb_confidence * age_match * breed_match * size_match, 2),
            fact_base["symptoms"],
            diagnosis["illness"],
            user_answers,
        )

        refined_diagnoses.append(
            {
                "illness": diagnosis["illness"],
                "matched_symptoms": diagnosis["matched_symptoms"],
                "confidence_fc": diagnosis["confidence_fc"],
                "confidence_gb": final_confidence_gb,
            }
        )

    return refined_diagnoses

def adaboost_ranking(possible_diagnoses, fact_base):
    """Uses AdaBoost to re-rank illnesses using model predictions and smart score blending."""
    user_symptoms = (
        [
            s["name"].lower().strip()
            for s in fact_base["symptoms"]
            if isinstance(s, dict) and "name" in s
        ]
        if isinstance(fact_base["symptoms"], list)
        else []
    )

    pet_info = fact_base["pet_info"]
    user_answers = fact_base["user_answers"]

    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # Step 1: Build full binary symptom vector
        feature_vector = pd.DataFrame(
            [
                [
                    1 if symptom.lower() in user_symptoms else 0
                    for symptom in all_symptoms
                ]
            ],
            columns=all_symptoms,
        )

        # Step 2: Add engineered scores
        fc_score = diagnosis["confidence_fc"]
        gb_score = diagnosis["confidence_gb"]
        match_ratio = len(diagnosis.get("matched_symptoms", [])) / max(
            len(user_symptoms), 1
        )

        feature_vector["FC_Confidence"] = fc_score
        feature_vector["GB_Confidence"] = gb_score
        feature_vector["Symptom_Match_Ratio"] = round(match_ratio, 4)

        # Step 3: Align to trained AdaBoost feature order
        feature_vector_ab = align_features(
            feature_vector, adaboost_model.feature_names_in_
        )

        # Step 4: Predict using AdaBoost
        raw_probs = adaboost_model.predict_proba(feature_vector_ab)
        predicted_label = adaboost_model.predict(feature_vector_ab)[0]
        illness_index = list(adaboost_model.classes_).index(diagnosis["illness"])
        ab_raw = raw_probs[0][illness_index]

        # Step 5: Boost score if AdaBoost also picked this illness
        ab_confidence = ab_raw
        if predicted_label == diagnosis["illness"]:
            ab_confidence += 0.2  # Small bonus if AdaBoost agrees

        # Step 6: Blended final confidence
        final_score = 0.2 * fc_score + 0.5 * gb_score + 0.3 * ab_confidence

        # Step 7: Apply pet info & follow-ups as modifiers
        rule = next(
            (r for r in kb_rules if r["illness"] == diagnosis["illness"]),
            None,
        )

        # Apply penalty
        penalty = 1.0

        if rule:
            # Age mismatch penalty
            rule_age = rule.get("age_range", "any").lower()
            pet_age = pet_info.get("age_range", "any").lower()
            if rule_age != "any" and rule_age != pet_age:
                penalty *= 0.7  # strong penalty for age mismatch

            # Breed mismatch penalty
            rule_breed = rule.get("breed", "any").strip().lower()
            pet_breed = pet_info.get("breed", "any").strip().lower()
            pet_breed_category = breed_category_mapping.get(
                pet_breed.title(), ""
            ).lower()
            if rule_breed != "any" and rule_breed not in [
                pet_breed,
                pet_breed_category,
            ]:
                penalty *= 0.85  # moderate penalty for breed mismatch

            # Size mismatch penalty
            rule_size = rule.get("size", "any").lower()
            pet_size = pet_info.get("size", "any").lower()
            if rule_size != "any" and rule_size != pet_size:
                penalty *= 0.85  # moderate penalty for size mismatch

        # Apply penalty before follow-up adjustments
        adjusted_score = round(final_score * penalty, 2)

        # Step 8: Apply follow-up refinements
        final_confidence_ab = adjust_confidence_with_followups(
            adjusted_score,
            fact_base["symptoms"],
            diagnosis["illness"],
            user_answers,
        )

        # Normalize each confidence score between 0 and 1
        normalized_fc = round(min(fc_score / 1.0, 1.0), 4)
        normalized_gb = round(min(gb_score / 1.0, 1.0), 4)
        normalized_ab = round(min(final_confidence_ab / 1.0, 1.0), 4)

        refined_diagnoses.append(
            {
                "illness": diagnosis["illness"],
                "matched_symptoms": diagnosis["matched_symptoms"],
                "confidence_fc": normalized_fc,
                "confidence_gb": normalized_gb,
                "confidence_ab": normalized_ab,
            }
        )

    return refined_diagnoses

def add_subtype_coverage_all(diagnoses, fact_base):
    """Add subtype coverage scores to all diagnoses"""
    for diagnosis in diagnoses:
        rule = next((r for r in kb_rules if r["illness"] == diagnosis["illness"]), None)
        diagnosis["subtype_coverage"] = compute_subtype_coverage(rule, fact_base["user_answers"]) if rule else 0.0
    return diagnoses

def apply_softmax_to_confidences(diagnoses):
    """Applies softmax to the final AdaBoost confidence scores of the diagnoses list."""
    scores = np.array([d["confidence_ab"] for d in diagnoses], dtype=float)
    exp_scores = np.exp(scores)
    softmax_probs = exp_scores / np.sum(exp_scores)

    for i, d in enumerate(diagnoses):
        # Format as decimal with exactly 4 decimal places
        d["confidence_softmax"] = round(softmax_probs[i], 4)


    return diagnoses

def build_comparison_output(diagnoses, fact_base):
    """Returns a structured comparison between top 2 illnesses."""
    if len(diagnoses) < 2:
        return {}

    illness_1, illness_2 = diagnoses[:2]
    illness_1_rule = next(
        (r for r in kb_rules if r["illness"] == illness_1["illness"]), None
    )
    illness_2_rule = next(
        (r for r in kb_rules if r["illness"] == illness_2["illness"]), None
    )
    if not illness_1_rule or not illness_2_rule:
        return {}

    coverage_1 = compute_subtype_coverage(illness_1_rule, fact_base["user_answers"])
    coverage_2 = compute_subtype_coverage(illness_2_rule, fact_base["user_answers"])

    return {
        "top_illness": illness_1["illness"],
        "second_illness": illness_2["illness"],
        "factors": [
            {
                "name": "Confidence Score",
                "top": illness_1["confidence_ab"],
                "second": illness_2["confidence_ab"],
            },
            {
                "name": "Weighted Symptom Matches",
                "top": illness_1["confidence_fc"],
                "second": illness_2["confidence_fc"],
            },
            {
                "name": "ML Score Adjustment",
                "top": round(
                    illness_1["confidence_ab"] - illness_1["confidence_fc"], 2
                ),
                "second": round(
                    illness_2["confidence_ab"] - illness_2["confidence_fc"], 2
                ),
            },
            {
                "name": "Subtype Coverage Score",
                "top": round(coverage_1, 2),
                "second": round(coverage_2, 2),
            },
        ],
        "reason_summary": {
            "why_top_ranked_higher": [
                f"Matched more weighted symptoms ({illness_1['confidence_fc']} vs {illness_2['confidence_fc']})",
                f"Better subtype alignment ({coverage_1}% vs {coverage_2}%)",
                "Machine learning still favored it after adjustments",
                "Fewer critical symptoms were missing",
            ]
        },
    }

def diagnose(fact_base):
    """Main diagnosis function with both top_10 and top_3 outputs."""
    # Define threshold for probability filteringr
    PROB_THRESHOLD = 0.04  # 4% minimum probability
    
    # Extract owner before normalization
    owner = fact_base.get("owner", "")
    
    # 1) Normalize payload
    fact_base = normalize_fact_base(fact_base)
    
    # Restore owner field
    fact_base["owner"] = owner
    
    # 2) Forward chaining
    fc = forward_chaining(fact_base)
    if not fc:
         return {
                    "owner":        owner,
                    "pet_info":     fact_base["pet_info"],
                    "symptoms":     fact_base["symptoms"],
                    "user_answers": fact_base["user_answers"],
                    "possible_diagnosis": [],
                    "top_diagnoses": [],
                    "allIllness": [],
                    "comparison": {}
                }
    
    # 3) Gradient Boosting → AdaBoost → subtype coverage
    gb = gradient_boosting_ranking(fc, fact_base)
    ab = adaboost_ranking(gb, fact_base)
    ab = add_subtype_coverage_all(ab, fact_base)
    
    # 4) SoftMax → sort descending
    ab = apply_softmax_to_confidences(ab)
    ab.sort(key=lambda d: d["confidence_softmax"], reverse=True)
    
    # 5) Set the possible_diagnosis in fact_base and save
    fact_base["possible_diagnosis"] = ab
    os.makedirs(os.path.dirname(FACT_BASE_PATH), exist_ok=True)
    with open(FACT_BASE_PATH, "w") as f:
        json.dump(fact_base, f, indent=4)
    
    # 6) Filter top results by threshold and limit to top 3
    filtered = [d for d in ab if d["confidence_softmax"] >= PROB_THRESHOLD]
    # Ensure at least one result if possible
    possible_diagnosis = filtered[:3] if filtered else ab[:1] if ab else []
    
    # 7) Build full top 10 with details
    top_diagnoses = []
    for d in ab:
        rule = next((r for r in kb_rules if r["illness"] == d["illness"]), {})
        info = illness_info_db.get(d["illness"], {})
        top_diagnoses.append({
            "illness":            d["illness"],
            "confidence_fc":      d["confidence_fc"],
            "confidence_gb":      d["confidence_gb"],
            "confidence_ab":      d["confidence_ab"],
            "confidence_softmax": d["confidence_softmax"],
            "subtype_coverage":   d.get("subtype_coverage", 0.0),
            "matched_symptoms":   d["matched_symptoms"],
            "missing_symptoms":   sorted(list(
                {s["name"].lower() for s in rule.get("symptoms", [])}
                - set(d["matched_symptoms"])
            )),
            "severity":       info.get("severity", "Unknown"),
            "description":    info.get("description", "No description available."),
            "treatment":      info.get("treatment", "No treatment guidelines provided."),
            "causes":         info.get("causes", "N/A"),
            "transmission":   info.get("transmission", "N/A"),
            "diagnosis":      info.get("diagnosis", "N/A"),
            "what_to_do":     info.get("what_to_do", "N/A"),
            "recovery_time":  info.get("recovery_time", "N/A"),
            "risk_factors":   info.get("risk_factors", []),
            "prevention":     info.get("prevention", "N/A"),
            "contagious":     info.get("contagious", False),
        })
    
    # 8) Build comparison block
    comparison = build_comparison_output(ab, fact_base)
    
    # 9) Return everything
    return {
                "owner":              owner,
                "pet_info":           fact_base["pet_info"],
                "symptoms":           fact_base["symptoms"],
                "user_answers":       fact_base["user_answers"],
                "possible_diagnosis": possible_diagnosis,
                "top_diagnoses":      top_diagnoses,  # fully built with info
                "allIllness":         ab,             # raw model output for diagnosis length
                "comparison":         comparison
            }


# ───── Evaluation Functions ─────

def evaluate_illness_performance(target_illness):
    """
    Evaluates the 2x2 confusion matrix and classification metrics
    for the given illness based on the full labeled dataset.
    """
    # Load true vs predicted results
    eval_path = "data/true_vs_predicted.csv"
    if not os.path.exists(eval_path):
        print("⚠️ Evaluation file not found.")
        return None

    df_eval = pd.read_csv(eval_path)

    # Binary classification: target illness vs all others
    y_true = df_eval["Illness"].apply(lambda x: 1 if x == target_illness else 0)
    y_pred = df_eval["Predicted"].apply(lambda x: 1 if x == target_illness else 0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

    return {
        "illness": target_illness,
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        },
        "metrics": {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Specificity": round(specificity, 4),
            "F1 Score": round(f1, 4),
        },
    }

def get_metrics_for_illness(illness_name):
    """Helper function to get metrics for a specific illness"""
    perf = evaluate_illness_performance(illness_name)
    if perf is None:
        return {}
    m = perf["metrics"]
    return {
        "accuracy": m["Accuracy"],
        "precision": m["Precision"],
        "recall": m["Recall"],
        "specificity": m["Specificity"],
        "f1": m["F1 Score"],
    }

# ───── FastAPI Endpoints ─────

@app.get("/metrics/{illness_name}")
async def read_metrics(illness_name: str):
    perf = evaluate_illness_performance(illness_name)
    if perf is None:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return perf["metrics"]

@app.get("/metrics-with-cm/{illness_name}")
async def read_metrics_with_cm(illness_name: str):
    perf = evaluate_illness_performance(illness_name)
    if perf is None:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return perf

@app.post("/diagnose")
async def diagnose_pet(request: Request):
    try:
        fact_base = await request.json()
        return diagnose(fact_base)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/all-symptoms")
async def get_all_symptoms():
    return {r["illness"]: [s["name"] for s in r["symptoms"]] for r in kb_rules}

@app.get("/debug/knowledge-details")
async def get_knowledge_details(illness: str):
    rule = next((r for r in kb_rules if r["illness"].lower()==illness.lower()), None)
    if not rule:
        raise HTTPException(status_code=404, detail="Illness not found")

    severity_multiplier = {"Low": 0.6, "Medium": 0.8, "High": 1.0, "Moderate": 0.8}
    processed = []
    total_ab_weight = 0.0
    num_symptoms = max(len(rule["symptoms"]), 1)

    for symptom in rule["symptoms"]:
        name = symptom["name"]
        base_weight = symptom.get("weight", 1.0)
        sev_mult = severity_multiplier.get(symptom.get("severity","Medium").capitalize(), 0.8)
        fc_weight = round(base_weight * sev_mult * symptom.get("priority",1.0), 2)

        key = name.lower().strip()
        idx = selected_features.index(key) if key in selected_features else None
        gb_importance = (boosting_model.feature_importances_[idx] if idx is not None else 0.3)
        gb_adjustment = round(fc_weight * gb_importance, 2)
        gb_weight = round(fc_weight + gb_adjustment, 2)

        ab_importance = (adaboost_model.feature_importances_[idx] if idx is not None and hasattr(adaboost_model, "feature_importances_") else 0.3)
        ab_factor = round(0.75 + (ab_importance * 0.1), 2)
        ab_weight = round(gb_weight * ab_factor, 2)

        total_ab_weight += ab_weight

        processed.append({
            "name": name,
            "base_weight": base_weight,
            "severity": symptom.get("severity","Medium"),
            "priority": symptom.get("priority",1.0),
            "fc_weight": fc_weight,
            "gb_adjustment": gb_adjustment,
            "gb_weight": gb_weight,
            "ab_factor": ab_factor,
            "ab_weight": ab_weight
        })

    final_confidence = round(min(100, (total_ab_weight/num_symptoms)*100), 2)
    return {"knowledge": processed, "final_confidence": final_confidence}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)