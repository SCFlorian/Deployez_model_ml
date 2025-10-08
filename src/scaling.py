import joblib
import pandas as pd
from src.utils import scaler_ou_non

def data_scaling(donnees_traitees):
    """
    Applique le scaler sauvegardé sur les colonnes numériques,
    et aligne les colonnes du DataFrame sur celles attendues par le modèle.
    """
    # Charger le scaler sauvegardé
    scaler = joblib.load("models/standard_scaler.pkl")

    # Récupérer les colonnes attendues
    features_a_scaler, features_encodees = scaler_ou_non()
    colonnes_attendues = features_a_scaler + features_encodees

    # Vérifie et ajoute les colonnes manquantes
    for col in colonnes_attendues:
        if col not in donnees_traitees.columns:
            donnees_traitees[col] = 0

    # Réordonne les colonnes dans le même ordre que lors de l'entraînement
    donnees_traitees = donnees_traitees[colonnes_attendues]

    # Appliquer le scaling sur les colonnes numériques uniquement
    donnees_traitees[features_a_scaler] = scaler.transform(donnees_traitees[features_a_scaler])

    return donnees_traitees