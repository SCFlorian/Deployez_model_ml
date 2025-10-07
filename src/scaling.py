import joblib
from src.utils import scaler_ou_non

def data_scaling(donnees_traitees):
    """
    Applique le scaler sauvegardé sur les colonnes numériques.
    """
    # Supprimer la colonne cible si elle existe
    if "a_quitte_l_entreprise" in donnees_traitees.columns:
        donnees_traitees = donnees_traitees.drop(columns=["a_quitte_l_entreprise"])

    # Charger le scaler sauvegardé
    scaler = joblib.load("models/standard_scaler.pkl")

    # Récupérer les colonnes à scaler
    features_a_scaler, _ = scaler_ou_non()

    # Vérification des colonnes présentes
    features_a_scaler = [col for col in features_a_scaler if col in donnees_traitees.columns]

    # Appliquer le scaling
    donnees_traitees[features_a_scaler] = scaler.transform(donnees_traitees[features_a_scaler])

    return donnees_traitees