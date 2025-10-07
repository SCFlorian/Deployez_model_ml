from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import gradio as gr

# Import des modules internes
from src.preprocessing import data_engineering
from src.scaling import data_scaling
from src.prediction import predict
from interface import build_interface


# Partie API FastAPI

class EmployeeInput(BaseModel):
    age: int
    genre: str
    revenu_mensuel: float
    statut_marital: str
    departement: str
    poste: str
    niveau_hierarchique_poste: int
    nombre_experiences_precedentes: int
    annee_experience_totale: int
    annees_dans_l_entreprise: int
    annees_dans_le_poste_actuel: int
    satisfaction_employee_environnement: float
    note_evaluation_precedente: float
    satisfaction_employee_nature_travail: float
    satisfaction_employee_equipe: float
    satisfaction_employee_equilibre_pro_perso: float
    note_evaluation_actuelle: float
    heure_supplementaires: str
    augmentation_salaire_precedente_pourcent: float
    nombre_participation_pee: int
    nb_formations_suivies: int
    distance_domicile_travail: float
    niveau_education: int
    domaine_etude: str
    frequence_deplacement: str
    annees_depuis_la_derniere_promotion: int
    annes_sous_responsable_actuel: int


fastapi_app = FastAPI(
    title="Employee Turnover Prediction API",
    description="API de prédiction du départ des employés basée sur un modèle de Machine Learning.",
    version="1.0.0"
)


@fastapi_app.get("/health")
def health_check():
    return {"status": "OK", "message": "API opérationnelle"}


@fastapi_app.post("/predict")
def predict_api(input_data: EmployeeInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        df = data_engineering(df)
        df = data_scaling(df)
        result = predict(df)
        message = "Risque de départ" if result["prediction"] == 1 else "Employé fidèle"
        return {
            "prediction": int(result["prediction"]),
            "probability": float(result["probability"]),
            "message": message
        }
    except Exception as e:
        return {"error": str(e)}


# Interface Gradio principale
demo = build_interface()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)