from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Text, ForeignKey, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from dotenv import load_dotenv

# === Connexion locale PostgreSQL ===

load_dotenv()  # Charge les variables depuis .env
DB_URL = os.getenv("DATABASE_URL")

# === Fallback automatique pour Hugging Face ===
if not DB_URL or "localhost" in DB_URL:
    print("⚠️  Aucun accès PostgreSQL détecté — utilisation de SQLite (mode Hugging Face).")
    DB_URL = "sqlite:///./default.db"

# === Connexion et session ===
connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

#  TABLE 1 : Données brutes (inputs du formulaire)

class EmployeeInputDB(Base):
    __tablename__ = "employee_inputs"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    genre = Column(String)
    revenu_mensuel = Column(Float)
    statut_marital = Column(String)
    departement = Column(String)
    poste = Column(String)
    niveau_hierarchique_poste = Column(Integer)
    nombre_experiences_precedentes = Column(Integer)
    annee_experience_totale = Column(Integer)
    annees_dans_l_entreprise = Column(Integer)
    annees_dans_le_poste_actuel = Column(Integer)
    satisfaction_employee_environnement = Column(Float)
    note_evaluation_precedente = Column(Float)
    satisfaction_employee_nature_travail = Column(Float)
    satisfaction_employee_equipe = Column(Float)
    satisfaction_employee_equilibre_pro_perso = Column(Float)
    note_evaluation_actuelle = Column(Float)
    heure_supplementaires = Column(String)
    augmentation_salaire_precedente_pourcent = Column(Float)
    nombre_participation_pee = Column(Integer)
    nb_formations_suivies = Column(Integer)
    distance_domicile_travail = Column(Float)
    niveau_education = Column(Integer)
    domaine_etude = Column(String)
    frequence_deplacement = Column(String)
    annees_depuis_la_derniere_promotion = Column(Integer)
    annes_sous_responsable_actuel = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    # Relations
    features = relationship("FeatureDB", back_populates="employee")
    predictions = relationship("PredictionResultDB", back_populates="employee")
    requests = relationship("RequestLogDB", back_populates="employee")

#  TABLE 2 : Données prêtes après preprocessing (features)

class FeatureDB(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, index=True)
    employee_input_id = Column(Integer, ForeignKey("employee_inputs.id"))
    feature_data = Column(Text)  # Données encodées/scalées au format JSON
    created_at = Column(DateTime, server_default=func.now())

    # Relation
    employee = relationship("EmployeeInputDB", back_populates="features")

#  TABLE 3 : Résultats de prédiction

class PredictionResultDB(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    employee_input_id = Column(Integer, ForeignKey("employee_inputs.id"))
    prediction = Column(Integer)
    probability = Column(Float)
    message = Column(String)
    created_at = Column(DateTime, server_default=func.now())

    # Relation
    employee = relationship("EmployeeInputDB", back_populates="predictions")

#  TABLE 4 : Journalisation des requêtes API

class RequestLogDB(Base):
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String)
    employee_input_id = Column(Integer, ForeignKey("employee_inputs.id"))
    user_id = Column(String, default="florian_user")
    timestamp = Column(DateTime, server_default=func.now())

    # Relations
    employee = relationship("EmployeeInputDB", back_populates="requests")
    responses = relationship("ApiResponseDB", back_populates="request")

#  TABLE 5 : Journalisation des réponses API

class ApiResponseDB(Base):
    __tablename__ = "api_responses"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("requests.id"))
    prediction_id = Column(Integer, ForeignKey("prediction_results.id"))
    status_code = Column(Integer)
    message = Column(String)
    timestamp = Column(DateTime, server_default=func.now())

    # Relations
    request = relationship("RequestLogDB", back_populates="responses")

#  CRÉATION DES TABLES

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Base de données et tables créées avec succès.")
