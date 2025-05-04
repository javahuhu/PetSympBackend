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
        msg["From"]    = EMAIL
        msg["To"]      = req.email
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

# ───── Diagnosis Core Functions ─────

def load_pet_resources(pet_type):
    if pet_type == "dog":
        kb = load_json("data/dog_knowledge_base.json")["rules"]
        boost_model = joblib.load("new model/new_dog_gradient_model.pkl")
        ada_model = joblib.load("new model/new_dog_adaboost_model.pkl")
        feats = joblib.load("new model/new_dog_adaboost_selected_features.pkl")
        dataset = pd.read_csv("data/dog_augmented.csv")
    else:
        kb = load_json("data/cat_knowledge_base.json")["rules"]
        boost_model = joblib.load("new model/new_cat_gradient_model.pkl")
        ada_model = joblib.load("new model/new_cat_adaboost_model.pkl")
        feats = joblib.load("new model/new_cat_adaboost_selected_features.pkl")
        dataset = pd.read_csv("data/cat_augmented.csv")
    return kb, boost_model, ada_model, feats, dataset

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

def adjust_confidence_with_followups(confidence,
                                     symptom_details,
                                     illness_name,
                                     user_answers,
                                     pet_type):
    """
    Modify confidence based on user follow-up answers.
    Chooses cat vs. dog impact map.
    """
    # pick the right map
    pet = pet_type.strip().lower()
    if pet == "dog":
        followups = DOG_FOLLOWUPS
    elif pet == "cat":
        followups = CAT_FOLLOWUPS
    else:
        followups = {}

    illness_rule = next((r for r in knowledge_base
                         if r["illness"] == illness_name), None)
    if not illness_rule:
        return confidence

    total_multiplier = 1.0
    for symptom in symptom_details:
        key = symptom.lower().strip()
        if key not in followups:
            continue

        cfg  = followups[key]
        impact_map = cfg.get("impact", {})
        user_resp  = user_answers.get(key, {})

        for answer in user_resp.values():
            norm = answer.strip().lower()
            if norm in impact_map:
                val = impact_map[norm]
                total_multiplier *= val
                print(f"✔ [{pet}] '{key}' → '{norm}': x{val}")

    # safety clamp
    total_multiplier = min(max(total_multiplier, 0.5), 2.0)
    return round(confidence * total_multiplier, 2)





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
            possible_diagnoses.append({
                "illness": rule["illness"],
                "matched_symptoms": [
                    s["name"].lower().strip()
                    for s in rule["symptoms"]
                    if s["name"].lower().strip() in user_symptoms
                ],
                "confidence_fc": confidence_fc,
            })
    return possible_diagnoses

def gradient_boosting_ranking(possible_diagnoses, fact_base):
    """Uses Gradient Boosting to refine illness ranking based on symptom match, pet info, and follow-up adjustments."""

    user_symptoms = [s.lower().strip() for s in fact_base["symptoms"]]
    pet_info = fact_base["pet_info"]
    user_answers = fact_base.get("user_answers", {})  # ✅ Safe default

    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # Build feature vector
        feature_vector = pd.DataFrame(
            [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms,
        )
        feature_vector_selected = align_features(feature_vector[selected_features], boosting_model.feature_names_in_)

        # Model prediction
        raw_predictions = boosting_model.predict_proba(feature_vector_selected)
        illness_index = list(boosting_model.classes_).index(diagnosis["illness"])
        gb_confidence = raw_predictions[0][illness_index]

        # Apply age, breed, size matching
        rule = next((r for r in knowledge_base if r["illness"] == diagnosis["illness"]), None)
        if rule:
            age_match = 1.1 if rule.get("age_range", "Any").lower() == pet_info.get("age_range", "Any").lower() else 0.85

            rule_breed = rule.get("breed", "Any").strip().lower()
            pet_breed = pet_info.get("breed", "Any").strip().lower()
            pet_breed_category = breed_category_mapping.get(pet_breed.title(), "").lower()

            if rule_breed == "any":
                breed_match = 1.0
            elif rule_breed == pet_breed or rule_breed == pet_breed_category:
                breed_match = 1.1
            else:
                breed_match = 0.9

            size_match = 1.1 if rule.get("size", "Any").lower() == pet_info.get("size", "Any").lower() else 0.9
        else:
            age_match = breed_match = size_match = 1.0

        initial_gb_confidence = round(gb_confidence * age_match * breed_match * size_match, 2)

        # ✅ Now dynamically adjust using follow-up answers (just like CLI)
        final_confidence_gb = adjust_confidence_with_followups(
            initial_gb_confidence,
            fact_base["symptoms"],
            diagnosis["illness"],
            user_answers,
            fact_base.get("pet_type", "dog")
        )

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": diagnosis["confidence_fc"],
            "confidence_gb": final_confidence_gb,
        })

    return refined_diagnoses



def adaboost_ranking(possible_diagnoses, fact_base):
    """Uses AdaBoost to re-rank illnesses using model predictions, smart blending, and follow-up dynamic adjustments."""

    user_symptoms = [s.lower().strip() for s in fact_base["symptoms"]]


    pet_info = fact_base["pet_info"]
    user_answers = fact_base.get("user_answers", {})

    refined_diagnoses = []

    for diagnosis in possible_diagnoses:
        # Build feature vector
        feature_vector = pd.DataFrame(
            [[1 if symptom.lower() in user_symptoms else 0 for symptom in all_symptoms]],
            columns=all_symptoms,
        )

        # Add engineered features
        fc_score = diagnosis["confidence_fc"]
        gb_score = diagnosis["confidence_gb"]
        match_ratio = len(diagnosis.get("matched_symptoms", [])) / max(len(user_symptoms), 1)

        feature_vector["FC_Confidence"] = fc_score
        feature_vector["GB_Confidence"] = gb_score
        feature_vector["Symptom_Match_Ratio"] = round(match_ratio, 4)

        feature_vector_ab = align_features(feature_vector, adaboost_model.feature_names_in_)

        # Predict AdaBoost
        raw_probs = adaboost_model.predict_proba(feature_vector_ab)
        predicted_label = adaboost_model.predict(feature_vector_ab)[0]
        illness_index = list(adaboost_model.classes_).index(diagnosis["illness"])
        ab_raw = raw_probs[0][illness_index]

        ab_confidence = ab_raw
        if predicted_label == diagnosis["illness"]:
            ab_confidence += 0.2  # Bonus if AdaBoost predicts it

        # Blend FC, GB, AB
        blended_score = 0.2 * fc_score + 0.5 * gb_score + 0.3 * ab_confidence

        # Apply penalties based on pet info mismatch
        penalty = 1.0
        rule = next((r for r in knowledge_base if r["illness"] == diagnosis["illness"]), None)
        illness_type = rule.get("type", "Unknown") if rule else "Unknown"
        if rule:
            rule_age = rule.get("age_range", "any").lower()
            rule_breed = rule.get("breed", "any").lower()
            rule_size = rule.get("size", "any").lower()

            pet_age = pet_info.get("age_range", "any").lower()
            pet_breed = pet_info.get("breed", "any").lower()
            pet_size = pet_info.get("size", "any").lower()

            pet_breed_category = breed_category_mapping.get(pet_breed.title(), "").lower()

            if rule_age != "any" and rule_age != pet_age:
                penalty *= 0.7  # Strong penalty for age mismatch
            if rule_breed != "any" and rule_breed not in [pet_breed, pet_breed_category]:
                penalty *= 0.85  # Moderate penalty for breed mismatch
            if rule_size != "any" and rule_size != pet_size:
                penalty *= 0.85  # Moderate penalty for size mismatch

        adjusted_score = round(blended_score * penalty, 2)

        # Apply dynamic follow-up adjustments
        final_confidence_ab = adjust_confidence_with_followups(
            adjusted_score,
            fact_base["symptoms"],
            diagnosis["illness"],
            user_answers,
            fact_base.get("pet_type", "dog")
        )

        normalized_fc = round(min(fc_score, 1.0), 4)
        normalized_gb = round(min(gb_score, 1.0), 4)
        normalized_ab = round(min(final_confidence_ab, 1.0), 4)

        refined_diagnoses.append({
            "illness": diagnosis["illness"],
            "matched_symptoms": diagnosis["matched_symptoms"],
            "confidence_fc": normalized_fc,
            "confidence_gb": normalized_gb,
            "confidence_ab": normalized_ab,
            "type": illness_type,
            "age_specificity": rule.get("age_range", "Any"),  
            "size_specificity": rule.get("size", "Any")    
        })

    return refined_diagnoses



def apply_softmax_to_confidences(diagnoses, threshold=0.0):
    scores = np.array([d["confidence_ab"] for d in diagnoses], dtype=float)
    exp_scores = np.exp(scores)
    softmax_probs = exp_scores / np.sum(exp_scores)

    for i, d in enumerate(diagnoses):
        d["confidence_softmax"] = round(float(softmax_probs[i]), 4)

    # 📌 If a threshold is set, filter illnesses after softmax
    if threshold > 0.02:
        diagnoses = [d for d in diagnoses if d["confidence_softmax"] >= threshold]

    return diagnoses

def compute_subtype_coverage(illness_rule, user_answers):
    """
    Computes a normalized subtype coverage score for an illness.
    """
    matched = 0
    total = 0
    for symptom in illness_rule.get("symptoms", []):
        symptom_name = symptom["name"].lower().strip()
        expected_subtypes = symptom.get("subtype", "")
        expected_list = [s.strip().lower() for s in expected_subtypes.split(",") if s.strip()]
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


# ───── Diagnose Endpoint ─────

@app.post("/diagnose")
async def diagnose_pet(request: Request):
    try:
        fact_base = await request.json()

        # 1) Get pet_type from the root of the JSON
        pet_type = fact_base.get("pet_type", "dog").lower()

        # 2) Load the appropriate KB and models
        global knowledge_base, boosting_model, adaboost_model, selected_features, df, all_symptoms
        knowledge_base, boosting_model, adaboost_model, selected_features, df = load_pet_resources(pet_type)
        all_symptoms = [col for col in df.columns if col != "Illness"]

        # 3) Ensure age_range is set, using "age" if present
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
                "comparison": {}
            }

        # Step 2: Gradient Boosting Ranking
        refined_diagnoses = gradient_boosting_ranking(possible_diagnoses, fact_base)

        # Step 3: AdaBoost Ranking
        final_diagnoses = adaboost_ranking(refined_diagnoses, fact_base)

        # Step 4: Compute subtype coverage
        for d in final_diagnoses:
            rule = next((r for r in knowledge_base if r["illness"] == d["illness"]), None)
            d["subtype_coverage"] = compute_subtype_coverage(rule, user_answers) if rule else 0.0

        # Step 5: Save raw AdaBoost confidence
        for d in final_diagnoses:
            d["confidence_ab_raw"] = d["confidence_ab"]

        # Step 6: Sort by raw AdaBoost confidence
        final_diagnoses.sort(key=lambda x: -x["confidence_ab_raw"])

        # Step 7: Apply softmax with threshold
        final_diagnoses = apply_softmax_to_confidences(final_diagnoses, threshold=0.02)

        # Step 8: Attach to fact_base
        fact_base["possible_diagnosis"] = final_diagnoses

        # Step 9: Persist fact_base
        os.makedirs(os.path.dirname(FACT_BASE_PATH), exist_ok=True)
        with open(FACT_BASE_PATH, "w") as f:
            json.dump(fact_base, f, indent=4)

        # Step 10: Build comparison for top 2
        comparison = {}
        if len(final_diagnoses) >= 2:
            ill1, ill2 = final_diagnoses[0], final_diagnoses[1]
            comparison = {
                "illness_1": ill1["illness"],
                "illness_2": ill2["illness"],
                "confidence_score": {
                    "illness_1": ill1["confidence_ab_raw"],
                    "illness_2": ill2["confidence_ab_raw"]
                },
                "weighted_symptoms": {
                    "illness_1": ill1["confidence_fc"],
                    "illness_2": ill2["confidence_fc"]
                },
                "ml_adjustment": {
                    "illness_1": round(ill1["confidence_ab_raw"] - ill1["confidence_fc"], 4),
                    "illness_2": round(ill2["confidence_ab_raw"] - ill2["confidence_fc"], 4),
                },
                "subtype_coverage": {
                    "illness_1": ill1.get("subtype_coverage", 0.0),
                    "illness_2": ill2.get("subtype_coverage", 0.0),
                }
            }

        # Step 11: Top 3 slice
        top_3_diagnoses = final_diagnoses[:3]

        # Step 12: Final response
        return {
            "owner": fact_base.get("owner", "Unknown"),
            "pet_info": fact_base.get("pet_info", {}),
            "symptoms": fact_base.get("symptoms", []),
            "top_diagnoses": top_3_diagnoses,
            "possible_diagnosis": final_diagnoses,
            "allIllness": final_diagnoses,
            "comparison": comparison,
            "fact_base": fact_base
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")



    

    

    

@app.get("/debug/all-symptoms")
async def get_all_symptoms(pet_type: str = "dog"):
    """Return all symptoms for the given pet type."""
    kb, _, _, _, _ = load_pet_resources(pet_type.lower())
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
   
    kb, boost_model, ada_model, feats, dataset = load_pet_resources(pet_type.lower())
    
    rule = next((r for r in kb if r["illness"].lower() == illness.lower()), None)
    if not rule:
        raise HTTPException(status_code=404, detail="Illness not found in knowledge base.")

    severity_multiplier = {"Low": 0.6, "Medium": 0.8, "High": 1.0, "Moderate": 0.8}
    processed = []
    total_ab_weight = 0.0
    num_symptoms = max(len(rule["symptoms"]), 1)

    for symptom in rule["symptoms"]:
        name = symptom["name"]
        base_weight = symptom.get("weight", 1.0)
        sev_mult = severity_multiplier.get(symptom.get("severity", "Medium").capitalize(), 0.8)
        fc_weight = round(base_weight * sev_mult * symptom.get("priority", 1.0), 2)

        key = name.lower().strip()
        idx = feats.index(key) if key in feats else None
        gb_importance = boosting_model.feature_importances_[idx] if idx is not None else 0.3
        gb_adjustment = round(fc_weight * gb_importance, 2)
        gb_weight = round(fc_weight + gb_adjustment, 2)

        ab_importance = ada_model.feature_importances_[idx] if idx is not None and hasattr(ada_model, "feature_importances_") else 0.3
        ab_factor = round(0.75 + (ab_importance * 0.1), 2)
        ab_weight = round(gb_weight * ab_factor, 2)

        total_ab_weight += ab_weight

        processed.append({
            "name": name,
            "base_weight": base_weight,
            "severity": symptom.get("severity", "Medium"),
            "priority": symptom.get("priority", 1.0),
            "fc_weight": fc_weight,
            "gb_adjustment": gb_adjustment,
            "gb_weight": gb_weight,
            "ab_factor": ab_factor,
            "ab_weight": ab_weight
        })

    final_confidence = round(min(100, (total_ab_weight / num_symptoms) * 100), 2)
    return {"knowledge": processed, "final_confidence": final_confidence}

    


# ───── Run App ─────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
