import joblib

model = joblib.load("adaboost_model.pkl")
print(model.feature_names_in_)
