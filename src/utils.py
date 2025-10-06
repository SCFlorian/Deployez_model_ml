import pandas as pd

def scaler_ou_non():
    """
    Retourne la liste des colonnes à scaler et des colonnes encodées.
    """
    features_a_scaler = [
        'revenu_mensuel','annees_dans_l_entreprise','satisfaction_employee_environnement',
        'note_evaluation_precedente','satisfaction_employee_nature_travail','satisfaction_employee_equipe',
        'satisfaction_employee_equilibre_pro_perso','note_evaluation_actuelle',
        'augmentation_salaire_precedente_pourcent','distance_domicile_travail','niveau_education',
        'annees_depuis_la_derniere_promotion','experience_externe','score_satisfaction',
        'augmentation_par_formation','pee_par_anciennete'
    ]
    features_encodees = [
        'genre','heure_supplementaires','frequence_deplacement','a_suivi_formation','tranche_age',
        'statut_marital_Celibataire','statut_marital_Divorce','statut_marital_Marie',
        'departement_Commercial','departement_Consulting','departement_RessourcesHumaines',
        'poste_AssistantdeDirection','poste_CadreCommercial','poste_Consultant','poste_DirecteurTechnique',
        'poste_Manager','poste_ReprésentantCommercial','poste_RessourcesHumaines','poste_SeniorManager',
        'poste_TechLead','promotion_recente','domaine_etude_Autre','domaine_etude_Entrepreunariat',
        'domaine_etude_InfraCloud','domaine_etude_Marketing','domaine_etude_RessourcesHumaines',
        'domaine_etude_TransformationDigitale'
    ]

    return features_a_scaler, features_encodees