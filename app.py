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

# --- Initialisation de la base ---
from database.create_db import Base, engine

print("Initialisation des tables si absentes...")
Base.metadata.create_all(bind=engine)
print("Tables prêtes.")


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
    1. Sauvegarde des données brutes dans la base
    2. Transformation et scaling
    3. Sauvegarde des features traitées
    4. Prédiction à partir des données sauvegardées
    5. Journalisation complète du résultat et de la réponse
    """
    db: Session = SessionLocal()
    try:
        # Sauvegarde des données brutes saisies
        new_input = EmployeeInputDB(**input_data.dict())
        db.add(new_input)
        db.commit()
        db.refresh(new_input)

        # Journalisation de la requête
        new_request = RequestLogDB(
            endpoint="/predict",
            employee_input_id=new_input.id,
            user_id="florian_user",
            timestamp=datetime.utcnow()
        )
        db.add(new_request)
        db.commit()
        db.refresh(new_request)

        # Transformation et scaling
        donnees_saisie = pd.DataFrame([input_data.dict()])
        donnees_traitees = data_engineering(donnees_saisie)
        donnees_pret = data_scaling(donnees_traitees)

        # Sauvegarde des données prêtes dans la table "features"
        donnees_pret_json = donnees_pret.to_json(orient="records")
        new_feature = FeatureDB(
            employee_input_id=new_input.id,
            feature_data=donnees_pret_json
        )
        db.add(new_feature)
        db.commit()
        db.refresh(new_feature)

        # Relecture propre pour prédiction (on relit les données sauvegardées)
        donnees_pret_reloaded = pd.read_json(new_feature.feature_data, orient="records")

        # Prédiction à partir des données rechargées
        result = predict(donnees_pret_reloaded)
        message = "Risque de départ" if result["prediction"] == 1 else "Employé fidèle"

        # Sauvegarde du résultat
        new_result = PredictionResultDB(
            employee_input_id=new_input.id,
            prediction=int(result["prediction"]),
            probability=float(result["probability"]),
            message=message
        )
        db.add(new_result)
        db.commit()
        db.refresh(new_result)

        # Journalisation de la réponse
        new_response = ApiResponseDB(
            request_id=new_request.id,
            prediction_id=new_result.id,
            status_code=200,
            message=message
        )
        db.add(new_response)
        db.commit()
        db.refresh(new_response)

        # 7On prépare les données à renvoyer avant fermeture de la session
        prediction_data = {
            "prediction": new_result.prediction,
            "probability": new_result.probability,
            "message": message
        }

        # Fermeture propre de la session
        db.close()

        # Envoi du résultat à l’utilisateur
        return prediction_data

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
