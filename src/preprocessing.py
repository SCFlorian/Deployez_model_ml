import pandas as pd
from sklearn.preprocessing import LabelEncoder

# FONCTION DATA ENGINEERING

def data_engineering(donnees_features: pd.DataFrame):
# Suppression de 3 colonnes redondantes par matrice de Pearson
    colonnes_a_supprimer = ['niveau_hierarchique_poste','annees_dans_le_poste_actuel','annes_sous_responsable_actuel']
    donnees_features = donnees_features.drop(columns = colonnes_a_supprimer)
# Plusieurs étapes de feature engineering
# Ajout de nouvelles variables
    donnees_features['experience_externe'] = donnees_features['annee_experience_totale'] - donnees_features['annees_dans_l_entreprise']
    
    donnees_features['score_satisfaction'] = (donnees_features[
    'satisfaction_employee_environnement'] + donnees_features['satisfaction_employee_nature_travail']
    + donnees_features['satisfaction_employee_equipe']+ donnees_features['satisfaction_employee_equilibre_pro_perso'])/4

    donnees_features['augmentation_par_formation'] = (donnees_features[
        'augmentation_salaire_precedente_pourcent']*100) / (donnees_features['nb_formations_suivies']+1)

    donnees_features['pee_par_anciennete'] = donnees_features['nombre_participation_pee'] / (donnees_features['annees_dans_l_entreprise']+1)
# Suppression de 3 colonnes redondantes par matrice de Spearman
    donnees_features = donnees_features.drop(columns='nombre_participation_pee')
    donnees_features = donnees_features.drop(columns='nombre_experiences_precedentes')
    donnees_features = donnees_features.drop(columns='annee_experience_totale')
# Modification des colonnes avec encodage
    donnees_features["a_suivi_formation"] = (donnees_features["nb_formations_suivies"] >= 1).astype(int)
    donnees_features = donnees_features.drop(columns='nb_formations_suivies')

    donnees_features['tranche_age'] = pd.cut(donnees_features[
        'age'],bins=[17, 30, 36, 43, 60],
        labels=['18-30', '31-36', '37-43','44+'])
    labelencoder = LabelEncoder()
    donnees_features['tranche_age']= labelencoder.fit_transform(donnees_features['tranche_age'])
    donnees_features = donnees_features.drop(columns='age')

    donnees_features['genre'] = donnees_features['genre'].map({'F': 1, 'M': 0})

    donnees_features = pd.get_dummies(donnees_features, columns=["statut_marital"], dtype=int)

    donnees_features = pd.get_dummies(donnees_features, columns=["departement"], dtype=int)

    donnees_features = pd.get_dummies(donnees_features, columns=["poste"], dtype=int)

    donnees_features = pd.get_dummies(donnees_features, columns=["domaine_etude"], dtype=int)

    donnees_features['heure_supplementaires'] = donnees_features['heure_supplementaires'].map({'Oui': 1, 'Non': 0})

    map_frequence = {"Aucun": 0, "Occasionnel": 1, "Frequent": 2}
    donnees_features["frequence_deplacement"] = donnees_features["frequence_deplacement"].map(map_frequence)

    donnees_features['promotion_recente'] = (donnees_features['annees_depuis_la_derniere_promotion'] <= 2).astype(int)

    # --- Patch de compatibilité pour le modèle entraîné ---
    if "a_quitte_l_entreprise" not in donnees_features.columns:
        donnees_features["a_quitte_l_entreprise"] = 0

    donnees_traitees = donnees_features.copy()

    return donnees_traitees