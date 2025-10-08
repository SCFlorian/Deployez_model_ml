from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# === Connexion locale PostgreSQL ===
# URL dans ma base de données
DB_URL = "postgresql://florianschorer@localhost:5432/employee_turnover"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# === Table des entrées utilisateur ===
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

# === Table des prédictions ===
class PredictionResultDB(Base):
    __tablename__ = "prediction_results"
    id = Column(Integer, primary_key=True, index=True)
    employee_input_id = Column(Integer)
    prediction = Column(Integer)
    probability = Column(Float)
    message = Column(String)
    created_at = Column(DateTime, server_default=func.now())

# === Création des tables ===
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Base de données et tables créées avec succès.")