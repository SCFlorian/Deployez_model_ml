from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import gradio as gr
import uvicorn

# Import des modules internes
from src.preprocessing import data_engineering
from src.scaling import data_scaling
from src.prediction import predict
from src.interface import build_interface


# === Schéma de validation Pydantic ===
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


# === Création de l’app FastAPI ===
app = FastAPI(
    title="Employee Turnover Prediction API",
    description="API de prédiction du départ des employés basée sur un modèle de Machine Learning.",
    version="1.0.0"
)


# === Endpoint de santé ===
@app.get("/health")
def health_check():
    """Vérifie si l’API fonctionne correctement."""
    return {"status": "OK", "message": "API opérationnelle"}


# === Endpoint de prédiction ===
@app.post("/predict")
def predict_api(input_data: EmployeeInput):
    """
    Endpoint principal : exécute le pipeline complet
    Feature engineering → Scaling → Prédiction
    """
    try:
        donnees_saisie = pd.DataFrame([input_data.dict()])
        donnees_traitees = data_engineering(donnees_saisie)
        donnees_pret = data_scaling(donnees_traitees)
        result = predict(donnees_pret)

        message = "Risque de départ" if result["prediction"] == 1 else "Employé fidèle"
        return {
            "prediction": int(result["prediction"]),
            "probability": float(result["probability"]),
            "message": message
        }

    except Exception as e:
        return {"error": str(e)}

# === Interface Gradio (UI) ===
demo = build_interface()

# === Montage de Gradio sur FastAPI ===
gradio_app = gr.mount_gradio_app(app, demo, path="/")

# === Lancement pour Docker ===
if __name__ == "__main__":
    uvicorn.run("app:gradio_app", host="0.0.0.0", port=7860)