import pandas as pd
import joblib

model = joblib.load("models/final_model.pkl")
with open("models/threshold.txt", "r") as f:
    threshold = float(f.read())

def predict(donnees_pret: pd.DataFrame):
    """
    Applique le modèle sur les données préparées et retourne la prédiction et la probabilité.
    """
    proba = model.predict_proba(donnees_pret)[:, 1][0]
    prediction = int(proba >= threshold)
    return {
        "prediction": prediction,
        "probability": round(float(proba), 4)
    }