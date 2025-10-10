from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import gradio as gr
import uvicorn
from sqlalchemy.orm import Session
from datetime import datetime

# === Import des modules internes ===
from src.preprocessing import data_engineering
from src.scaling import data_scaling
from src.prediction import predict
from src.interface import build_interface
from database.create_db import (
    SessionLocal,
    EmployeeInputDB,
    PredictionResultDB,
    FeatureDB,
    RequestLogDB,
    ApiResponseDB,
)

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


# === Création de l’application FastAPI ===
app = FastAPI(
    title="Employee Turnover Prediction API",
    description="API de prédiction du départ des employés avec journalisation complète et sécurité renforcée.",
    version="2.0.0"
)

# === Endpoint de santé ===
@app.get("/health")
def health_check():
    """Vérifie si l’API fonctionne et journalise la requête."""
    db = SessionLocal()

    new_request = RequestLogDB(
        endpoint="/health",
        user_id="florian_user",
        timestamp=datetime.utcnow()
    )
    db.add(new_request)
    db.commit()
    db.close()

    return {"status": "OK", "message": "API opérationnelle"}


# === Endpoint principal de prédiction ===
@app.post("/predict")
def predict_api(input_data: EmployeeInput):
    """
    Étapes :
    1. Enregistrement des données brutes
    2. Transformation & scaling
    3. Enregistrement des features transformées
    4. Prédiction
    5. Enregistrement du résultat et journalisation complète
    6. Diffusion du résultat à l’utilisateur
    """
    db: Session = SessionLocal()
    try:
        # Journaliser la requête et sauvegarder les données brutes
        new_input = EmployeeInputDB(**input_data.dict())
        db.add(new_input)
        db.commit()
        db.refresh(new_input)

        new_request = RequestLogDB(
            endpoint="/predict",
            employee_input_id=new_input.id,
            user_id="florian_user",
            timestamp=datetime.utcnow()
        )
        db.add(new_request)
        db.commit()

        # Transformation et scaling
        donnees_saisie = pd.DataFrame([input_data.dict()])
        donnees_traitees = data_engineering(donnees_saisie)
        donnees_pret = data_scaling(donnees_traitees)

        # Sauvegarde des données prêtes avant prédiction
        donnees_pret_json = donnees_pret.to_json(orient="records")
        new_feature = FeatureDB(
            employee_input_id=new_input.id,
            feature_data=donnees_pret_json
        )
        db.add(new_feature)
        db.commit()

        donnees_pret_reloaded = pd.read_json(new_feature.feature_data, orient="records")

        # Prédiction via le modèle (à partir des données relues)
        result = predict(donnees_pret_reloaded)
        message = "Risque de départ" if result["prediction"] == 1 else "Employé fidèle"

        # Sauvegarde + relecture du résultat
        new_result = PredictionResultDB(
            employee_input_id=new_input.id,
            prediction=result["prediction"],
            probability=result["probability"],
            message=message
        )
        db.add(new_result)
        db.commit()

        prediction_record = db.query(PredictionResultDB).filter_by(id=new_result.id).first()

        # Journalisation de la réponse
        new_response = ApiResponseDB(
            request_id=new_request.id,
            prediction_id=prediction_record.id,
            status_code=200,
            message=prediction_record.message
        )
        db.add(new_response)
        db.commit()

        # Fermeture propre + retour utilisateur
        db.close()

        return {
            "prediction": int(prediction_record.prediction),
            "probability": float(prediction_record.probability),
            "message": prediction_record.message
        }

    except Exception as e:
        db.rollback()
        db.close()
        return {"error": str(e)}


# === Interface Gradio (UI) ===
demo = build_interface()
gradio_app = gr.mount_gradio_app(app, demo, path="/")


# === Lancement local ===
if __name__ == "__main__":
    uvicorn.run("app:gradio_app", host="0.0.0.0", port=7860)
