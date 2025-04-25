import joblib

# Load the selected features list
selected_features = joblib.load("adaboost_selected_features.pkl")

print("\n🔍 Total Features:", len(selected_features))
print("🔝 Sample Feature Names:")
for f in selected_features[:10]:
    print(" -", f)

# Look for any engineered features
print("\n🔎 Engineered Features (suspicious):")
suspicious = [
    f for f in selected_features if "confidence" in f.lower() or "ratio" in f.lower()
]
if suspicious:
    for f in suspicious:
        print("⚠️", f)
else:
    print("✅ None found — feature list is clean.")
