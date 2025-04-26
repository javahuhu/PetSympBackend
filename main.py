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
import firebase_admin
from firebase_admin import auth, credentials
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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
FACT_BASE_PATH = "data/fact_base.json"
ILLNESS_INFO_PATH = "data/expanded_illness_info_complete.json"
FOLLOWUP_QUESTIONS_PATH = "data/updated_follow_up_questions_tuned.json"
BREED_CATEGORY_MAP_PATH = "data/breed_category_mapping.json"

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

# ───── Helper Functions from First Script ─────
def categorize_age(age):
    try:
        age = int(age)
    except ValueError:
        return "Unknown"
    if age <= 1:
        return "Puppy"
    elif 1 < age <= 7:
        return "Adult"
    else:
        return "Senior"

def validate_size(sz):
    return sz.capitalize() if isinstance(sz, str) and sz.lower() in ["small","medium","large"] else "Medium"

def parse_duration_range(range_str):
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
    user_interval = parse_duration_range(user_range_str)
    expected_interval = parse_duration_range(expected_range_str)
    if user_interval and expected_interval:
        return max(user_interval[0], expected_interval[0]) <= min(user_interval[1], expected_interval[1])
    return False

def align_features(feature_vector, expected_features):
    feature_vector = feature_vector.copy()
    for feature in expected_features:
        if feature not in feature_vector.columns:
            feature_vector.loc[:, feature] = 0
    return feature_vector[expected_features]

def compute_subtype_coverage(rule, user_answers):
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
                user_duration = user_response.get(f"How long has your pet had {symptom_name}?", None)
                user_severity = user_response.get(f"Is the {symptom_name} Mild, Moderate, or Severe?", None)

                user_subtype = None
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

                impact_values = symptom_followups.get(symptom_name, {}).get("impact", {})
                severity_impact = impact_values.get(user_severity.lower(), 1.2) if user_severity else 1.2
                subtype_impact = impact_values.get(user_subtype_clean, 1.2) if user_subtype_clean else 1.2

                if user_duration and expected_duration.lower() != "any":
                    if duration_overlap(user_duration, expected_duration):
                        duration_impact = impact_values.get(user_duration.lower(), 1.2)
                    else:
                        duration_impact = 0.95
                else:
                    duration_impact = 1.2

                kb_match_bonus = 1.0
                if user_severity and expected_severity.lower() != "any":
                    kb_match_bonus *= 1.02 if user_severity.lower() == expected_severity.lower() else 0.95
                if user_duration and expected_duration.lower() != "any":
                    kb_match_bonus *= 1.03 if duration_overlap(user_duration, expected_duration) else 0.95
                if user_subtype_clean and expected_subtypes != ["any"]:
                    kb_match_bonus *= 1.08 if user_subtype_clean in expected_subtypes else 0.9

                total_multiplier *= (severity_impact * subtype_impact * duration_impact * kb_match_bonus)
    return round(confidence * total_multiplier, 2)

def forward_chaining(fact_base):
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
            possible_diagnoses.append({
                "illness": rule["illness"],
                "matched_symptoms": [
                    s["name"].lower().strip()
                    for s in rule["symptoms"]
                    if s["name"].lower().strip() in user_symptoms
                ],
                "confidence_fc": confidence_fc
            })
    return possible_diagnoses

def gradient_boosting_ranking(possible_diagnoses, fact_base):
    user_symptoms = [
        s["name"].lower().strip()
        for s in fact_base["symptoms"]
        if isinstance(s, dict) and "name" in s
    ]
    pet_info = fact_base["pet_info"]
    user_answers = fact_base["user_answers"]
    refined = []
    for diag in possible_diagnoses:
        fv = pd.DataFrame([[1 if s in user_symptoms else 0 for s in all_symptoms]], columns=all_symptoms)
        fv_sel = align_features(fv, boosting_model.feature_names_in_)
        raw_preds = boosting_model.predict_proba(fv_sel)
        idx = list(boosting_model.classes_).index(diag["illness"])
        gb_conf = raw_preds[0][idx]
        rule = next((r for r in kb_rules if r["illness"] == diag["illness"]), None)
        if rule:
            age_match = 1.1 if rule.get("age_range","Any").lower()==pet_info.get("age_range","Any").lower() else 0.85
            rb, pb = rule.get("breed","Any").strip().lower(), pet_info.get("breed","Any").strip().lower()
            bc = breed_category_mapping.get(pet_info.get("breed","Any").title(),"").lower()
            breed_match = 1.0 if rb=="any" else 1.1 if rb in [pb,bc] else 0.9
            size_match = 1.1 if rule.get("size","Any").lower()==pet_info.get("size","Any").lower() else 0.9
        else:
            age_match=breed_match=size_match=1.0
        final_gb = adjust_confidence_with_followups(
            round(gb_conf*age_match*breed_match*size_match,2),
            fact_base["symptoms"], diag["illness"], user_answers
        )
        refined.append({
            "illness": diag["illness"],
            "matched_symptoms": diag["matched_symptoms"],
            "confidence_fc": diag["confidence_fc"],
            "confidence_gb": final_gb
        })
    return refined

def adaboost_ranking(possible_diagnoses, fact_base):
    user_symptoms = [
        s["name"].lower().strip()
        for s in fact_base["symptoms"]
        if isinstance(s, dict) and "name" in s
    ]
    pet_info = fact_base["pet_info"]
    user_answers = fact_base["user_answers"]
    refined = []
    for diag in possible_diagnoses:
        fv = pd.DataFrame([[1 if s in user_symptoms else 0 for s in all_symptoms]], columns=all_symptoms)
        fc_sc, gb_sc = diag["confidence_fc"], diag["confidence_gb"]
        match_ratio = len(diag["matched_symptoms"])/max(len(user_symptoms),1)
        fv["FC_Confidence"], fv["GB_Confidence"], fv["Symptom_Match_Ratio"] = fc_sc, gb_sc, round(match_ratio,4)
        fv_ab = align_features(fv, adaboost_model.feature_names_in_)
        raw_probs = adaboost_model.predict_proba(fv_ab)
        pred_lbl = adaboost_model.predict(fv_ab)[0]
        idx = list(adaboost_model.classes_).index(diag["illness"])
        ab_raw = raw_probs[0][idx]
        ab_conf = ab_raw + (0.2 if pred_lbl==diag["illness"] else 0)
        final_score = 0.2*fc_sc + 0.5*gb_sc + 0.3*ab_conf
        rule = next((r for r in kb_rules if r["illness"]==diag["illness"]),None)
        penalty = 1.0
        if rule:
            ra,pa = rule.get("age_range","any").lower(), pet_info.get("age_range","any").lower()
            if ra!="any" and ra!=pa: penalty*=0.7
            rb,pb = rule.get("breed","any").strip().lower(), pet_info.get("breed","any").strip().lower()
            bc = breed_category_mapping.get(pet_info.get("breed","any").title(),"").lower()
            if rb!="any" and rb not in [pb,bc]: penalty*=0.85
            rs,ps = rule.get("size","any").lower(), pet_info.get("size","any").lower()
            if rs!="any" and rs!=ps: penalty*=0.85
        adjusted = round(final_score*penalty,2)
        final_ab = adjust_confidence_with_followups(adjusted, fact_base["symptoms"], diag["illness"], user_answers)
        nfc = round(min(fc_sc/1.0,1.0),4)
        ngb = round(min(gb_sc/1.0,1.0),4)
        nab = round(min(final_ab/1.0,1.0),4)
        refined.append({
            "illness": diag["illness"],
            "matched_symptoms": diag["matched_symptoms"],
            "confidence_fc": nfc,
            "confidence_gb": ngb,
            "confidence_ab": nab
        })
    return refined

def add_subtype_coverage_all(diagnoses, fact_base):
    for d in diagnoses:
        rule = next((r for r in kb_rules if r["illness"]==d["illness"]),None)
        d["subtype_coverage"] = compute_subtype_coverage(rule, fact_base["user_answers"]) if rule else 0.0
    return diagnoses

def apply_softmax_to_confidences(diagnoses):
    scores = np.array([d["confidence_ab"] for d in diagnoses],dtype=float)
    exp_scores = np.exp(scores)
    sm = exp_scores/np.sum(exp_scores)
    for i,d in enumerate(diagnoses):
        d["confidence_softmax"] = round(sm[i],4)
    return diagnoses

def build_comparison_output(diagnoses, fact_base):
    if len(diagnoses)<2: return {}
    d1,d2 = diagnoses[0],diagnoses[1]
    r1 = next((r for r in kb_rules if r["illness"]==d1["illness"]),None)
    r2 = next((r for r in kb_rules if r["illness"]==d2["illness"]),None)
    if not r1 or not r2: return {}
    c1 = compute_subtype_coverage(r1,fact_base["user_answers"])
    c2 = compute_subtype_coverage(r2,fact_base["user_answers"])
    return {
        "top_illness": d1["illness"],
        "second_illness": d2["illness"],
        "factors": [
            {"name":"Confidence Score","top":d1["confidence_ab"],"second":d2["confidence_ab"]},
            {"name":"Weighted Symptom Matches","top":d1["confidence_fc"],"second":d2["confidence_fc"]},
            {"name":"ML Score Adjustment","top":round(d1["confidence_ab"]-d1["confidence_fc"],2),"second":round(d2["confidence_ab"]-d2["confidence_fc"],2)},
            {"name":"Subtype Coverage Score","top":round(c1,2),"second":round(c2,2)}
        ],
        "reason_summary": {"why_top_ranked_higher":[
            f"Matched more weighted symptoms ({d1['confidence_fc']} vs {d2['confidence_fc']})",
            f"Better subtype alignment ({c1}% vs {c2}%)",
            "Machine learning still favored it after adjustments",
            "Fewer critical symptoms were missing"
        ]}
    }

def normalize_fact_base(fb):
    pi = fb.get("pet_info",{})
    age = pi.get("age_years") or pi.get("age")
    if age is not None:
        pi["age_years"] = str(age)
        if "age" in pi: del pi["age"]
    pi["age_range"] = categorize_age(pi.get("age_years","0"))
    pi["size"] = validate_size(pi.get("size","Medium"))
    pi["breed"] = pi.get("breed","Any").strip().capitalize()
    fb["pet_info"] = pi
    syms = fb.get("symptoms",[])
    norm = []
    for s in syms:
        if isinstance(s,str):
            norm.append({"name":s.lower().strip()})
        elif isinstance(s,dict) and "name" in s:
            norm.append({"name":s["name"].lower().strip()})
    fb["symptoms"] = norm
    ua = fb.get("user_answers",{})
    nua={}
    for sym,ans in ua.items():
        key = sym.lower().strip()
        if isinstance(ans,dict):
            nua[key] = {q.strip():a for q,a in ans.items()}
    fb["user_answers"] = nua
    return fb

def evaluate_illness_performance(target_illness):
    eval_path = "data/true_vs_predicted.csv"
    if not os.path.exists(eval_path):
        return None
    df_eval = pd.read_csv(eval_path)
    y_true = df_eval["Illness"].apply(lambda x:1 if x==target_illness else 0)
    y_pred = df_eval["Predicted"].apply(lambda x:1 if x==target_illness else 0)
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    precision = precision_score(y_true,y_pred,zero_division=0)
    recall    = recall_score(y_true,y_pred,zero_division=0)
    f1        = f1_score(y_true,y_pred,zero_division=0)
    specificity = tn/(tn+fp) if (tn+fp)!=0 else 0
    accuracy    = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)!=0 else 0
    return {
        "illness": target_illness,
        "confusion_matrix": {"TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn)},
        "metrics": {
            "Accuracy":round(accuracy,4),
            "Precision":round(precision,4),
            "Recall":round(recall,4),
            "Specificity":round(specificity,4),
            "F1 Score":round(f1,4)
        }
    }

def get_metrics_for_illness(illness_name):
    perf = evaluate_illness_performance(illness_name)
    if perf is None:
        return {}
    m = perf["metrics"]
    return {"accuracy":m["Accuracy"],"precision":m["Precision"],"recall":m["Recall"],"specificity":m["Specificity"],"f1":m["F1 Score"]}

# ───── Core Diagnose Function & Endpoint ─────
def diagnose(fact_base):
    owner = fact_base.get("owner","")
    fact_base = normalize_fact_base(fact_base)
    fact_base["owner"] = owner
    fc = forward_chaining(fact_base)
    if not fc:
        return {
            "owner":owner,
            "pet_info":fact_base["pet_info"],
            "symptoms":fact_base["symptoms"],
            "user_answers":fact_base["user_answers"],
            "possible_diagnosis":[],
            "top_diagnoses":[],
            "allIllness":[],
            "comparison":{}
        }
    gb = gradient_boosting_ranking(fc,fact_base)
    ab = adaboost_ranking(gb,fact_base)
    ab = add_subtype_coverage_all(ab,fact_base)
    ab = apply_softmax_to_confidences(ab)
    ab.sort(key=lambda d: d["confidence_softmax"], reverse=True)
    fact_base["possible_diagnosis"] = ab
    os.makedirs(os.path.dirname(FACT_BASE_PATH), exist_ok=True)
    with open(FACT_BASE_PATH,"w") as f:
        json.dump(fact_base, f, indent=4)
    PROB_THRESHOLD = 0.04
    filtered = [d for d in ab if d["confidence_softmax"]>=PROB_THRESHOLD]
    possible = filtered[:3] if filtered else ab[:1] if ab else []
    top_diagnoses = []
    for d in ab:
        rule = next((r for r in kb_rules if r["illness"]==d["illness"]),{})
        info = illness_info_db.get(d["illness"],{})
        top_diagnoses.append({
            "illness": d["illness"],
            "confidence_fc": d["confidence_fc"],
            "confidence_gb": d["confidence_gb"],
            "confidence_ab": d["confidence_ab"],
            "confidence_softmax": d["confidence_softmax"],
            "subtype_coverage": d.get("subtype_coverage",0.0),
            "matched_symptoms": d["matched_symptoms"],
            "missing_symptoms": sorted(list({s["name"].lower() for s in rule.get("symptoms",[])}-set(d["matched_symptoms"]))),
            "severity": info.get("severity","Unknown"),
            "description": info.get("description","No description available."),
            "treatment": info.get("treatment","No treatment guidelines provided."),
            "causes": info.get("causes","N/A"),
            "transmission": info.get("transmission","N/A"),
            "diagnosis": info.get("diagnosis","N/A"),
            "what_to_do": info.get("what_to_do","N/A"),
            "recovery_time": info.get("recovery_time","N/A"),
            "risk_factors": info.get("risk_factors",[]),
            "prevention": info.get("prevention","N/A"),
            "contagious": info.get("contagious",False)
        })
    comparison = build_comparison_output(ab,fact_base)
    return {
        "owner":owner,
        "pet_info":fact_base["pet_info"],
        "symptoms":fact_base["symptoms"],
        "user_answers":fact_base["user_answers"],
        "possible_diagnosis":possible,
        "top_diagnoses":top_diagnoses,
        "allIllness":ab,
        "comparison":comparison
    }

@app.post("/diagnose")
async def diagnose_pet(request: Request):
    try:
        fact_base = await request.json()
        return diagnose(fact_base)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ───── Debug Endpoints ─────
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
    num_symptoms = max(len(rule["symptoms"]),1)

    for symptom in rule["symptoms"]:
        name = symptom["name"]
        base_weight = symptom.get("weight",1.0)
        sev_mult = severity_multiplier.get(symptom.get("severity","Medium").capitalize(),0.8)
        fc_weight = round(base_weight * sev_mult * symptom.get("priority",1.0),2)
        idx = selected_features.index(name.lower().strip()) if name.lower().strip() in selected_features else None
        gb_importance = boosting_model.feature_importances_[idx] if idx is not None else 0.3
        gb_adjustment = round(fc_weight * gb_importance,2)
        gb_weight = round(fc_weight + gb_adjustment,2)
        ab_importance = adaboost_model.feature_importances_[idx] if idx is not None and hasattr(adaboost_model,"feature_importances_") else 0.3
        ab_factor = round(0.75 + (ab_importance*0.1),2)
        ab_weight = round(gb_weight * ab_factor,2)
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

    final_confidence = round(min(100,(total_ab_weight/num_symptoms)*100),2)
    return {"knowledge": processed, "final_confidence": final_confidence}

# ───── Run App ─────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
