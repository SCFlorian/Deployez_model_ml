import joblib
from src.utils import scaler_ou_non

def data_scaling(donnees_traitees):
    """
    Applique le scaler sauvegardé sur les colonnes numériques.
    """
    # Charger le scaler sauvegardé
    scaler = joblib.load("models/standard_scaler.pkl")

    # Récupérer les colonnes à scaler
    features_a_scaler, _ = scaler_ou_non()

    # Appliquer le scaling
    donnees_traitees[features_a_scaler] = scaler.transform(donnees_traitees[features_a_scaler])

    return donnees_traitees