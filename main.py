# main.py

import os
import json
import joblib
import uvicorn
import smtplib
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
from dotenv import load_dotenv
import base64


load_dotenv()
# ───── Initialization ─────
if not firebase_admin._apps:
    import base64
    encoded_key = os.environ.get('GOOGLE_CREDENTIALS')
    if not encoded_key:
        raise RuntimeError("GOOGLE_CREDENTIALS environment variable not set.")
    decoded_key = base64.b64decode(encoded_key)
    cred = credentials.Certificate(json.loads(decoded_key))
    firebase_admin.initialize_app(cred)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───── Static Paths ─────
FACT_BASE_PATH = "data/fact_base.json"
ILLNESS_INFO_PATH = "data/expanded_illness_info_complete.json"
FOLLOWUP_QUESTIONS_PATH = "data/updated_follow_up_questions_tuned.json"
BREED_CATEGORY_MAP_PATH = "data/breed_category_mapping.json"


# ───── Static Loads ─────
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


illness_info_db = load_json(ILLNESS_INFO_PATH)
breed_category_mapping = load_json(BREED_CATEGORY_MAP_PATH)
CAT_FOLLOWUPS = load_json("data/cat_follow_up.json")
DOG_FOLLOWUPS = load_json("data/dog_follow_up.json")
BOOST_PENALTY_RULES_PATH = "data/illness_boost_penalty_rules.json"
boost_penalty_rules = load_json(BOOST_PENALTY_RULES_PATH)

knowledge_base = []
boosting_model = None
adaboost_model = None
selected_features = []
df = pd.DataFrame()
all_symptoms = []

# ───── OTP System ─────
OTP_STORE = {}
EMAIL = "petsymp0@gmail.com"
APP_PASSWORD = "gqox rtam taom hhbb"


class OTPRequest(BaseModel):
    email: str
    otp: str


@app.post("/send-otp")
async def send_otp(req: OTPRequest):
    try:
        OTP_STORE[req.email] = req.otp
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your PetSymp OTP Code"
        msg["From"] = EMAIL
        msg["To"] = req.email
        html = f"""
        <html><body>
          <h2>PetSymp Email Verification</h2>
          <h3>Your OTP is:</h3>
          <h1><strong style='color:#52AAA4'>{req.otp}</strong></h1>
          <b>Important: For your security, never share your OTP with anyone.</b>
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
        raise HTTPException(
            status_code=400, detail="Email, new password, and OTP are required."
        )
    if OTP_STORE.get(email) != otp:
        raise HTTPException(status_code=401, detail="Invalid OTP.")
    try:
        user = auth.get_user_by_email(email)
        auth.update_user(user.uid, password=newpw)
        del OTP_STORE[email]
        return {"message": "Password updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update password: {e}")


# ───── Diagnosis Core Functions ─────


def categorize_age(age, pet_type="dog"):
    try:
        age = float(age)
    except ValueError:
        return "Unknown"
    if pet_type == "dog":
        return "Puppy" if age < 1 else "Adult" if age <= 7 else "Senior"
    else:
        return "Kitten" if age < 1 else "Adult" if age <= 10 else "Senior"


def align_features(feature_vector, expected_features):
    feature_vector = feature_vector.copy()
    for feature in expected_features:
        if feature not in feature_vector.columns:
            feature_vector[feature] = 0
    return feature_vector[expected_features]


def adjust_confidence_with_followups(confidence, symptom_details, illness_name, user_answers, pet_type):
    pet = pet_type.strip().lower()
    followups = DOG_FOLLOWUPS if pet == "dog" else CAT_FOLLOWUPS if pet == "cat" else {}

    illness_rule = next((r for r in knowledge_base if r["illness"] == illness_name), None)
    if not illness_rule:
        return round(min(confidence, 1.0), 4)

    total_multiplier = 1.0

    for symptom in symptom_details:
        key = symptom.lower().strip()
        if key not in followups:
            continue

        cfg = followups[key]
        impact_map = cfg.get("impact", {})
        user_resp = user_answers.get(key, {})

        multipliers = []
        for answer in user_resp.values():
            norm = answer.strip().lower()
            if norm in impact_map:
                val = impact_map[norm]
                multipliers.append(val)

        if multipliers:
            avg_impact = sum(multipliers) / len(multipliers)
            total_multiplier *= avg_impact

    # ✨ Step 4: Apply dynamic boost/penalty from illness_boost_penalty_rules.json
    rule_cfg = boost_penalty_rules.get(illness_name.lower(), {})

    # Dynamic boost
    if "boost_if_contains" in rule_cfg:
        for symptom_answers in user_answers.values():
            for val in symptom_answers.values():
                if any(keyword in val.lower() for keyword in rule_cfg["boost_if_contains"]):
                    total_multiplier = max(total_multiplier, rule_cfg.get("boost_factor", 1.5))

    # Dynamic penalty
    if "penalty_if_contains" in rule_cfg:
        for symptom_answers in user_answers.values():
            for val in symptom_answers.values():
                if any(keyword in val.lower() for keyword in rule_cfg["penalty_if_contains"]):
                    total_multiplier = min(total_multiplier, rule_cfg.get("penalty_factor", 0.7))

    total_multiplier = min(max(total_multiplier, 0.5), 3.0)
    return round(min(confidence * total_multiplier, 1.0), 4)

def normalize_user_answers(user_answers):
    return {
        k.lower().strip(): {
            q.lower().strip(): v.lower().strip()
            for q, v in v.items()
        } for k, v in user_answers.items()
    }


def forward_chaining(fact_base):
    user_symptoms = [s.lower().strip() for s in fact_base["symptoms"]]

    possible_diagnoses = []
    for rule in knowledge_base:
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


def gradient_boosting_ranking(
    possible_diagnoses, fact_base, boosting_model, selected_features, all_symptoms
):
    user_symptoms = [s.lower().strip() for s in fact_base["symptoms"]]
    pet_info = fact_base["pet_info"]
    user_answers = fact_base.get("user_answers", {})

    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # Build feature vector
        feature_vector = pd.DataFrame(
            [
                [
                    1 if symptom.lower() in user_symptoms else 0
                    for symptom in all_symptoms
                ]
            ],
            columns=all_symptoms,
        )

        # ✅ Only pass symptom features to GB
        feature_vector_selected = align_features(
            feature_vector, boosting_model.feature_names_in_
        )

        # Model prediction
        raw_predictions = boosting_model.predict_proba(feature_vector_selected)
        illness_index = list(boosting_model.classes_).index(diagnosis["illness"])
        gb_confidence = raw_predictions[0][illness_index]

        # ✅ Retrieve rule safely first
        rule = next(
            (r for r in knowledge_base if r["illness"] == diagnosis["illness"]), None
        )

        # ✅ Pattern Ratio Boost for GB (after rule is loaded)
        rule_symptom_count = len(rule.get("symptoms", [])) if rule else 0
        pattern_ratio = len(diagnosis["matched_symptoms"]) / max(rule_symptom_count, 1)
        boost_factor = 0.5 + 1.5 * pattern_ratio  # ranges from 0.5 to 2.0
        gb_confidence *= boost_factor
        gb_confidence = min(gb_confidence, 1.0)

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
            elif rule_breed == pet_breed or rule_breed == pet_breed_category:
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

        initial_gb_confidence = round(
            gb_confidence * age_match * breed_match * size_match, 2
        )

        # Adjust based on follow-up answers
        final_confidence_gb = adjust_confidence_with_followups(
            initial_gb_confidence,
            fact_base["symptoms"],
            diagnosis["illness"],
            user_answers,
            fact_base.get("pet_type", "dog"),
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


def adaboost_ranking(
    possible_diagnoses, fact_base, adaboost_model, selected_features, all_symptoms
):
    user_symptoms = [s.lower().strip() for s in fact_base["symptoms"]]

    pet_info = fact_base["pet_info"]
    user_answers = fact_base.get("user_answers", {})
    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # Build feature vector
        feature_vector = pd.DataFrame(
            [
                [
                    1 if symptom.lower() in user_symptoms else 0
                    for symptom in all_symptoms
                ]
            ],
            columns=all_symptoms,
        )
        fc_score = diagnosis["confidence_fc"]
        gb_score = diagnosis["confidence_gb"]
        match_ratio = len(diagnosis.get("matched_symptoms", [])) / max(
            len(user_symptoms), 1
        )

        # ✅ Add extra features
        feature_vector["FC_Confidence"] = fc_score
        feature_vector["GB_Confidence"] = gb_score
        feature_vector["Symptom_Match_Ratio"] = round(match_ratio, 4)
        # ✅ Then align to AB model
        feature_vector_ab = align_features(
            feature_vector, adaboost_model.feature_names_in_
        )

        # Predict AdaBoost
        raw_probs = adaboost_model.predict_proba(feature_vector_ab)
        predicted_label = adaboost_model.predict(feature_vector_ab)[0]
        illness_index = list(adaboost_model.classes_).index(diagnosis["illness"])
        ab_raw = raw_probs[0][illness_index]
        ab_confidence = ab_raw
        if predicted_label == diagnosis["illness"]:
            ab_confidence += 0.2
        blended_score = 0.2 * fc_score + 0.5 * gb_score + 0.3 * ab_confidence

        penalty = 1.0
        rule = next(
            (r for r in knowledge_base if r["illness"] == diagnosis["illness"]), None
        )
        illness_type = rule.get("type", "Unknown") if rule else "Unknown"

        if rule:
            # Pattern Ratio Boost
            total_illness_symptoms = len(rule.get("symptoms", []))
            pattern_ratio = len(diagnosis["matched_symptoms"]) / max(
                total_illness_symptoms, 1
            )
            blended_score *= 0.7 + 0.3 * pattern_ratio

            # Subtype Match Boost
            subtype_matched = 0
            subtype_total = 0
            for sym in rule.get("symptoms", []):
                name = sym["name"].lower().strip()
                subtypes = [
                    s.strip().lower() for s in sym.get("subtype", "").split(",") if s
                ]
                if not subtypes:
                    continue
                subtype_total += 1
                user_resp = user_answers.get(name, {})
                for ans in user_resp.values():
                    if ans.strip().lower() in subtypes:
                        subtype_matched += 1
                        break
            subtype_ratio = subtype_matched / subtype_total if subtype_total > 0 else 0
            subtype_boost = 0.75 + 0.25 * subtype_ratio  # increase effect slightly
            blended_score *= subtype_boost

            # ✅ Apply additional penalties based on profile mismatch
            rule_age = rule.get("age_range", "any").lower()
            rule_breed = rule.get("breed", "any").lower()
            rule_size = rule.get("size", "any").lower()
            pet_age = pet_info.get("age_range", "any").lower()
            pet_breed = pet_info.get("breed", "any").lower()
            pet_size = pet_info.get("size", "any").lower()
            pet_breed_category = breed_category_mapping.get(
                pet_breed.title(), ""
            ).lower()

            # ❗ Stronger penalty if age does not match
            if rule_age != "any" and rule_age != pet_age:
                penalty *= 0.5  # was 0.7
            # ❗ Stricter breed mismatch penalty
            if rule_breed != "any" and rule_breed not in [
                pet_breed,
                pet_breed_category,
            ]:
                penalty *= 0.7  # was 0.85
            # ❗ Stricter size mismatch penalty
            if rule_size != "any" and rule_size != pet_size:
                penalty *= 0.7  # was 0.85
        adjusted_score = round(blended_score * penalty, 2)

        final_confidence_ab = adjust_confidence_with_followups(
            adjusted_score,
            fact_base["symptoms"],
            diagnosis["illness"],
            user_answers,
            fact_base.get("pet_type", "dog"),
        )

        normalized_fc = round(min(fc_score, 1.0), 4)
        normalized_gb = round(min(gb_score, 1.0), 4)
        normalized_ab = round(min(final_confidence_ab, 1.0), 4)

        refined_diagnoses.append(
            {
                "illness": diagnosis["illness"],
                "matched_symptoms": diagnosis["matched_symptoms"],
                "confidence_fc": normalized_fc,
                "confidence_gb": normalized_gb,
                "confidence_ab": normalized_ab,
                "type": illness_type,
                "age_specificity": rule.get("age_range", "Any"),
                "size_specificity": rule.get("size", "Any"),
                "subtype_coverage": round(subtype_ratio * 100, 2),
            }
        )

    return refined_diagnoses


def apply_softmax_to_confidences(diagnoses, threshold=0.0):
    if not diagnoses:
        return []

    # Sort by AdaBoost raw confidence
    diagnoses.sort(key=lambda x: -x["confidence_ab"])

    # Take top 3 for softmax
    top_k = 3
    top_diagnoses = diagnoses[:top_k]
    rest = diagnoses[top_k:]

    # Apply softmax to top 3
    scores = np.array([d["confidence_ab"] for d in top_diagnoses], dtype=float)
    exp_scores = np.exp(scores)
    softmax_probs = exp_scores / np.sum(exp_scores)

    for i, d in enumerate(top_diagnoses):
        d["confidence_softmax"] = round(float(softmax_probs[i]), 4)

    # Assign 0.0 to the rest (or skip entirely if preferred)
    for d in rest:
        d["confidence_softmax"] = 0.0

    all_diagnoses = top_diagnoses + rest

    # Optional: filter only top-k if threshold is enabled
    if threshold > 0.0:
        all_diagnoses = [
            d for d in all_diagnoses if d["confidence_softmax"] >= threshold or d not in top_diagnoses
        ]

    return all_diagnoses



def compute_subtype_coverage(illness_rule, user_answers):
    """
    Computes a normalized subtype coverage score for an illness.
    """
    matched = 0
    total = 0
    for symptom in illness_rule.get("symptoms", []):
        symptom_name = symptom["name"].lower().strip()
        expected_subtypes = symptom.get("subtype", "")
        expected_list = [
            s.strip().lower() for s in expected_subtypes.split(",") if s.strip()
        ]
        if expected_list:
            total += 1
            user_response = user_answers.get(symptom_name, {})
            for answer in user_response.values():
                if answer.lower().strip() in expected_list:
                    matched += 1
                    break
    if total == 0:
        return 0.0
    return round((matched / total) * 100, 2)


# Utility to get unique illness types from top-N FC diagnoses
def get_detected_categories(possible_diagnoses, knowledge_base, top_n=3):
    detected_types = set()
    for diag in possible_diagnoses[:top_n]:
        illness = diag["illness"]
        rule = next((r for r in knowledge_base if r["illness"] == illness), None)
        if rule:
            detected_types.add(rule.get("type", "Unknown").lower())
    return detected_types


# Load models per category
CATEGORY_MODEL_PATHS = {
    "viral": {
        "gb": "{pet_type}_viral.pkl",
        "ab": "{pet_type}_viral.pkl",
        "features": "{pet_type}_viral_selected_features.pkl",
        "dataset": "{pet_type}_viral.csv",
    },
    "bacterial": {
        "gb": "{pet_type}_bacterial.pkl",
        "ab": "{pet_type}_bacterial.pkl",
        "features": "{pet_type}_bacterial_selected_features.pkl",
        "dataset": "{pet_type}_bacterial.csv",
    },
    "parasitic": {
        "gb": "{pet_type}_parasitic.pkl",
        "ab": "{pet_type}_parasitic.pkl",
        "features": "{pet_type}_parasitic_selected_features.pkl",
        "dataset": "{pet_type}_parasitic.csv",
    },
    "musculoskeletal": {
        "gb": "{pet_type}_musculoskeletal.pkl",
        "ab": "{pet_type}_musculoskeletal.pkl",
        "features": "{pet_type}_musculoskeletal_selected_features.pkl",
        "dataset": "{pet_type}_musculoskeletal.csv",
    },
    "cardiovascular": {
        "gb": "{pet_type}_cardiovascular.pkl",
        "ab": "{pet_type}_cardiovascular.pkl",
        "features": "{pet_type}_cardiovascular_selected_features.pkl",
        "dataset": "{pet_type}_cardiovascular.csv",
    },
    "neurological": {
        "gb": "{pet_type}_neurological.pkl",
        "ab": "{pet_type}_neurological.pkl",
        "features": "{pet_type}_neurological_selected_features.pkl",
        "dataset": "{pet_type}_neurological.csv",
    },
    "renal": {
        "gb": "{pet_type}_renal.pkl",
        "ab": "{pet_type}_renal.pkl",
        "features": "{pet_type}_renal_selected_features.pkl",
        "dataset": "{pet_type}_renal.csv",
    },
    "dental": {
        "gb": "{pet_type}_dental.pkl",
        "ab": "{pet_type}_dental.pkl",
        "features": "{pet_type}_dental_selected_features.pkl",
        "dataset": "{pet_type}_dental.csv",
    },
}


# Load models and data for a given category
def load_category_models(category, pet_type="dog"):
    try:
        paths = CATEGORY_MODEL_PATHS[category]
        gb_path = f"new gb model/{paths['gb'].format(pet_type=pet_type)}"
        ab_path = f"new ab model/{paths['ab'].format(pet_type=pet_type)}"
        features_path = f"new ab model/{paths['features'].format(pet_type=pet_type)}"
        dataset_path = f"new dataset/{paths['dataset'].format(pet_type=pet_type)}"

        gb = joblib.load(gb_path)
        ab = joblib.load(ab_path)
        feats = joblib.load(features_path)
        df = pd.read_csv(dataset_path)

        print(f"✅ Loaded models for {pet_type}-{category}")
        return gb, ab, feats, df

    except Exception as e:
        raise RuntimeError(f"Failed to load models for {pet_type}-{category}: {e}")


# ───── Diagnose Endpoint ─────


@app.post("/diagnose")
async def diagnose_pet(request: Request):
    try:
        fact_base = await request.json()
        pet_type = fact_base.get("pet_type", "dog").lower()

        # ✅ Sanity check for supported pet types
        if pet_type not in ["dog", "cat"]:
            raise HTTPException(
                status_code=400, detail="Invalid pet type. Must be 'dog' or 'cat'."
            )

        # Load base KB (used by FC and for rule lookups only)
        global knowledge_base
        knowledge_base = load_json(f"data/{pet_type}_knowledge_base.json")["rules"]

        if "pet_info" in fact_base:
            pi = fact_base["pet_info"]
            if "age_range" not in pi:
                raw_age = pi.get("age") or pi.get("age_years", 1)
                pi["age_range"] = categorize_age(raw_age, pet_type)

        user_answers = fact_base.get("user_answers", {})

        # Step 1: Forward Chaining
        possible_diagnoses = forward_chaining(fact_base)
        if not possible_diagnoses:
            return {
                "owner": fact_base.get("owner", "Unknown"),
                "pet_info": fact_base.get("pet_info", {}),
                "symptoms": fact_base.get("symptoms", []),
                "top_diagnoses": [],
                "possible_diagnosis": [],
                "allIllness": [],
                "comparison": {},
            }

        # Step 2: Detect illness categories from FC results
        categories = get_detected_categories(possible_diagnoses, knowledge_base)
        all_ranked_diagnoses = []

        for category in categories:
            try:
                boosting_model, adaboost_model, selected_features, df = (
                    load_category_models(category, pet_type)
                )
                print(f"✅ Loaded models for category: {category}")
                all_symptoms = [col for col in df.columns if col != "Illness"]

                # Filter FC illnesses for this category
                category_diagnoses = []
                for diag in possible_diagnoses:
                    rule = next(
                        (r for r in knowledge_base if r["illness"] == diag["illness"]),
                        None,
                    )
                    if rule and rule.get("type", "").lower() == category:
                        category_diagnoses.append(diag)

                # Step 3a: GB Ranking per category
                gb_ranked = gradient_boosting_ranking(
                    category_diagnoses,
                    fact_base,
                    boosting_model,
                    selected_features,
                    all_symptoms,
                )

                # Step 3b: AB Ranking per category
                ab_ranked = adaboost_ranking(
                    gb_ranked,
                    fact_base,
                    adaboost_model,
                    selected_features,
                    all_symptoms,
                )

                all_ranked_diagnoses.extend(ab_ranked)

            except Exception as e:
                print(f"⚠️ Skipping category '{category}': {e}")
                continue

        # Step 4: Add subtype coverage and store raw AB confidence
        for d in all_ranked_diagnoses:
            rule = next(
                (r for r in knowledge_base if r["illness"] == d["illness"]), None
            )
            d["subtype_coverage"] = (
                compute_subtype_coverage(rule, user_answers) if rule else 0.0
            )
            d["confidence_ab_raw"] = d["confidence_ab"]

        # Step 5: Sort by raw AB, apply softmax, and attach
        all_ranked_diagnoses.sort(key=lambda x: -x["confidence_ab_raw"])
        all_ranked_diagnoses = apply_softmax_to_confidences(
            all_ranked_diagnoses, threshold=0.00
        )
        fact_base["possible_diagnosis"] = all_ranked_diagnoses

        # Step 6: Save diagnosis to JSON
        os.makedirs(os.path.dirname(FACT_BASE_PATH), exist_ok=True)
        with open(FACT_BASE_PATH, "w") as f:
            json.dump(fact_base, f, indent=4)

        # Step 7: Top 2 comparison data
        comparison = {}
        if len(all_ranked_diagnoses) >= 2:
            ill1, ill2 = all_ranked_diagnoses[0], all_ranked_diagnoses[1]
            comparison = {
                "illness_1": ill1["illness"],
                "illness_2": ill2["illness"],
                "confidence_score": {
                    "illness_1": ill1["confidence_ab_raw"],
                    "illness_2": ill2["confidence_ab_raw"],
                },
                "weighted_symptoms": {
                    "illness_1": ill1["confidence_fc"],
                    "illness_2": ill2["confidence_fc"],
                },
                "ml_adjustment": {
                    "illness_1": round(
                        ill1["confidence_ab_raw"] - ill1["confidence_fc"], 4
                    ),
                    "illness_2": round(
                        ill2["confidence_ab_raw"] - ill2["confidence_fc"], 4
                    ),
                },
                "subtype_coverage": {
                    "illness_1": ill1.get("subtype_coverage", 0.0),
                    "illness_2": ill2.get("subtype_coverage", 0.0),
                },
            }

        return {
            "owner": fact_base.get("owner", "Unknown"),
            "pet_info": fact_base.get("pet_info", {}),
            "symptoms": fact_base.get("symptoms", []),
            "top_diagnoses": all_ranked_diagnoses[:3],
            "possible_diagnosis": all_ranked_diagnoses,
            "allIllness": all_ranked_diagnoses,
            "comparison": comparison,
            "fact_base": fact_base,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")


@app.get("/debug/all-symptoms")
async def get_all_symptoms(pet_type: str = "dog"):
    """Return all symptoms for the given pet type."""
    kb = load_json(f"data/{pet_type.lower()}_knowledge_base.json")["rules"]
    return {r["illness"]: [s["name"] for s in r["symptoms"]] for r in kb}


# ───── Evaluation Endpoints ─────


def evaluate_illness_performance(target_illness, pet_type="dog"):

    if pet_type.lower().strip() == "dog":
        eval_path = "data/dog_true_vs_predicted.csv"
    else:
        eval_path = "data/cat_true_vs_predicted.csv"

    if not os.path.exists(eval_path):
        print("Evaluation file not found.")
        return None

    df_eval = pd.read_csv(eval_path)

    # Normalize Illness and Predicted columns
    df_eval["Illness"] = df_eval["Illness"].astype(str).str.strip().str.lower()
    df_eval["Predicted"] = df_eval["Predicted"].astype(str).str.strip().str.lower()

    target_clean = target_illness.strip().lower()

    y_true = df_eval["Illness"].apply(lambda x: 1 if x == target_clean else 0)
    y_pred = df_eval["Predicted"].apply(lambda x: 1 if x == target_clean else 0)

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

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


@app.get("/metrics/{illness_name}")
async def read_metrics(illness_name: str, pet_type: str = "dog"):
    perf = evaluate_illness_performance(illness_name, pet_type)
    if perf is None:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return perf["metrics"]


@app.get("/metrics-with-cm/{illness_name}")
async def read_metrics_with_cm(illness_name: str, pet_type: str = "dog"):
    perf = evaluate_illness_performance(illness_name, pet_type)
    if perf is None:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return perf


@app.get("/debug/knowledge-details")
async def get_knowledge_details(pet_type: str, illness: str):
    # Load KB
    kb = load_json(f"data/{pet_type.lower()}_knowledge_base.json")["rules"]
    global knowledge_base
    knowledge_base = kb  # ensure consistency

    # Find the illness rule
    rule = next((r for r in kb if r["illness"].lower() == illness.lower()), None)
    if not rule:
        raise HTTPException(
            status_code=404, detail="Illness not found in knowledge base."
        )

    # Get the illness type to load proper category models
    illness_type = rule.get("type", "unknown").lower()
    try:
        boost_model, ada_model, feats, dataset = load_category_models(illness_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")

    # Continue as normal
    severity_multiplier = {"Low": 0.6, "Medium": 0.8, "High": 1.0, "Moderate": 0.8}
    processed = []
    total_ab_weight = 0.0
    num_symptoms = max(len(rule["symptoms"]), 1)

    for symptom in rule["symptoms"]:
        name = symptom["name"]
        base_weight = symptom.get("weight", 1.0)
        sev_mult = severity_multiplier.get(
            symptom.get("severity", "Medium").capitalize(), 0.8
        )
        fc_weight = round(base_weight * sev_mult * symptom.get("priority", 1.0), 2)

        key = name.lower().strip()
        idx = feats.index(key) if key in feats else None
        gb_importance = (
            boost_model.feature_importances_[idx] if idx is not None else 0.3
        )
        gb_adjustment = round(fc_weight * gb_importance, 2)
        gb_weight = round(fc_weight + gb_adjustment, 2)

        ab_importance = ada_model.feature_importances_[idx] if idx is not None else 0.3
        ab_factor = round(0.75 + (ab_importance * 0.1), 2)
        ab_weight = round(gb_weight * ab_factor, 2)

        total_ab_weight += ab_weight

        processed.append(
            {
                "name": name,
                "base_weight": base_weight,
                "severity": symptom.get("severity", "Medium"),
                "priority": symptom.get("priority", 1.0),
                "fc_weight": fc_weight,
                "gb_adjustment": gb_adjustment,
                "gb_weight": gb_weight,
                "ab_factor": ab_factor,
                "ab_weight": ab_weight,
            }
        )

    final_confidence = round(min(100, (total_ab_weight / num_symptoms) * 100), 2)
    return {"knowledge": processed, "final_confidence": final_confidence}


# ───── Run App ─────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)