import gradio as gr
import pandas as pd
import joblib
import requests
import os

# Fonctions de test
def test_api_health():
    """Teste si l’API FastAPI répond via le endpoint /health."""
    try:
        response = requests.get(f"{API_URL.replace('/predict', '')}/health")
        if response.status_code == 200:
            return f"API opérationnelle : {response.json().get('message', '')}"
        else:
            return f"API non disponible — Code {response.status_code}"
    except Exception as e:
        return f"Erreur de connexion à l’API — {e}"

def test_scaling():
    """Test du scaler (charge le scaler et vérifie les colonnes)."""
    try:
        joblib.load("models/standard_scaler.pkl")
        return "Endpoint 'Scaling' : Scaler chargé avec succès."
    except Exception as e:
        return f"Endpoint 'Scaling' : Erreur — {e}"

def test_model():
    """Test du chargement du modèle et du seuil."""
    try:
        joblib.load("models/final_model.pkl")
        with open("models/threshold.txt", "r") as f:
            threshold = float(f.read())
        return f"Endpoint 'Model' : Modèle et seuil ({threshold:.3f}) chargés avec succès."
    except Exception as e:
        return f"Endpoint 'Model' : Erreur — {e}"

# === Connexion à l’API FastAPI ===
SPACE_URL = os.getenv("SPACE_URL", "").strip().replace("_", "-").lower().rstrip("/")
API_URL = f"{SPACE_URL}/predict" if SPACE_URL else "http://localhost:7860/predict"

# Fonction principale de prédiction
def process_input(
    age, genre, revenu_mensuel, statut_marital,
    departement, poste, niveau_hierarchique_poste,
    nombre_experiences_precedentes, annee_experience_totale,
    annees_dans_l_entreprise, annees_dans_le_poste_actuel,
    satisfaction_employee_environnement, note_evaluation_precedente,
    satisfaction_employee_nature_travail, satisfaction_employee_equipe,
    satisfaction_employee_equilibre_pro_perso, note_evaluation_actuelle,
    heure_supplementaires, augmentation_salaire_precedente_pourcent,
    nombre_participation_pee, nb_formations_suivies,
    distance_domicile_travail, niveau_education,
    domaine_etude, frequence_deplacement,
    annees_depuis_la_derniere_promotion, annes_sous_responsable_actuel
):
    """Envoie les données saisies à l'API FastAPI pour prédiction."""
    input_data = {
        "age": age,
        "genre": genre,
        "revenu_mensuel": revenu_mensuel,
        "statut_marital": statut_marital,
        "departement": departement,
        "poste": poste,
        "niveau_hierarchique_poste": niveau_hierarchique_poste,
        "nombre_experiences_precedentes": nombre_experiences_precedentes,
        "annee_experience_totale": annee_experience_totale,
        "annees_dans_l_entreprise": annees_dans_l_entreprise,
        "annees_dans_le_poste_actuel": annees_dans_le_poste_actuel,
        "satisfaction_employee_environnement": satisfaction_employee_environnement,
        "note_evaluation_precedente": note_evaluation_precedente,
        "satisfaction_employee_nature_travail": satisfaction_employee_nature_travail,
        "satisfaction_employee_equipe": satisfaction_employee_equipe,
        "satisfaction_employee_equilibre_pro_perso": satisfaction_employee_equilibre_pro_perso,
        "note_evaluation_actuelle": note_evaluation_actuelle,
        "heure_supplementaires": heure_supplementaires,
        "augmentation_salaire_precedente_pourcent": augmentation_salaire_precedente_pourcent,
        "nombre_participation_pee": nombre_participation_pee,
        "nb_formations_suivies": nb_formations_suivies,
        "distance_domicile_travail": distance_domicile_travail,
        "niveau_education": niveau_education,
        "domaine_etude": domaine_etude,
        "frequence_deplacement": frequence_deplacement,
        "annees_depuis_la_derniere_promotion": annees_depuis_la_derniere_promotion,
        "annes_sous_responsable_actuel": annes_sous_responsable_actuel,
    }

    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            return f"Erreur API : {data['error']}"
            
        elif "message" in data and "probability" in data:
            return f"{data['message']} — probabilité : {data['probability']:.3f}"
        
        else:
            return f"Réponse inattendue du serveur : {data}"
    
    except Exception as e:
        return f"Erreur : {e}"

# Interface Gradio
def build_interface():
    """Construit l'interface Gradio."""
    with gr.Blocks(
        title="Employee Turnover Prediction",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="gray", neutral_hue="gray")
    ) as demo:
        gr.Markdown("# Prédiction du Départ des Employés")
        gr.Markdown("### Vérifiez les endpoints puis effectuez une prédiction complète.")

        # Section test des endpoints
        with gr.Row():
            test_output1 = gr.Textbox(label="Statut de l’API (Health Check)", interactive=False)
            gr.Button("Tester l’API").click(fn=test_api_health, outputs=test_output1)
            test_output2 = gr.Textbox(label="Résultat Scaling", interactive=False)
            gr.Button("Tester Scaling").click(fn=test_scaling, outputs=test_output2)
            test_output3 = gr.Textbox(label="Résultat Modèle", interactive=False)
            gr.Button("Tester Modèle").click(fn=test_model, outputs=test_output3)

        gr.Markdown("---")

        # Formulaire de prédiction
        gr.Markdown("## Saisir les informations de l’employé")

        with gr.Row():
            age = gr.Number(label="Âge", minimum=18, maximum=60, step=1)
            genre = gr.Radio(choices=["M", "F"], label="Genre")
            revenu = gr.Number(label="Revenu mensuel (€)", minimum=1000, maximum=20000)

        statut_marital = gr.Dropdown(["Celibataire", "Marie", "Divorce"], label="Statut marital")
        departement = gr.Dropdown(["Commercial", "Consulting", "RessourcesHumaines"], label="Département")
        poste = gr.Dropdown(["AssistantdeDirection","CadreCommercial","Consultant","DirecteurTechnique",
                             "Manager","ReprésentantCommercial","RessourcesHumaines","SeniorManager","TechLead"], label="Poste")

        niveau_hierarchique_poste = gr.Number(label="Niveau hiérarchique", minimum=1, maximum=5)
        nombre_experiences_precedentes = gr.Number(label="Expériences précédentes", minimum=1, maximum=9)
        annee_experience_totale = gr.Number(label="Années d'expérience totale", minimum=0, maximum=40)
        annees_dans_l_entreprise = gr.Number(label="Années dans l’entreprise", minimum=0, maximum=40)
        annees_dans_le_poste_actuel = gr.Number(label="Années dans le poste actuel", minimum=0, maximum=18)
        annees_depuis_la_derniere_promotion = gr.Number(label="Années depuis la dernière promotion", minimum=0, maximum=15)
        annes_sous_responsable_actuel = gr.Number(label="Années sous responsable actuel", minimum=0, maximum=17)

        satisfaction_employee_environnement = gr.Slider(1, 4, step=1, label="Satisfaction environnement")
        satisfaction_employee_nature_travail = gr.Slider(1, 4, step=1, label="Satisfaction nature du travail")
        satisfaction_employee_equipe = gr.Slider(1, 4, step=1, label="Satisfaction équipe")
        satisfaction_employee_equilibre_pro_perso = gr.Slider(1, 4, step=1, label="Satisfaction équilibre vie")
        note_evaluation_precedente = gr.Slider(1, 4, step=1, label="Évaluation précédente")
        note_evaluation_actuelle = gr.Slider(3, 4, step=1, label="Évaluation actuelle")

        heure_supplementaires = gr.Radio(["Oui", "Non"], label="Heures supplémentaires")
        augmentation_salaire_precedente_pourcent = gr.Dropdown(choices=[
            "0.11","0.12","0.13","0.14","0.15","0.16","0.17","0.18","0.19","0.20","0.21","0.22","0.23","0.24","0.25"
        ], label="Augmentation du salaire précédente (%)")
        nombre_participation_pee = gr.Number(label="Participations PEE", minimum=0, maximum=3, step=1)
        nb_formations_suivies = gr.Number(label="Formations suivies", minimum=0, maximum=6, step=1)
        distance_domicile_travail = gr.Number(label="Distance domicile-travail (km)", minimum=1, maximum=29, step=1)
        niveau_education = gr.Slider(1, 5, step=1, label="Niveau d'éducation")
        domaine_etude = gr.Dropdown(["Autre","Entrepreunariat","InfraCloud","Marketing","RessourcesHumaines","TransformationDigitale"], label="Domaine d’étude")
        frequence_deplacement = gr.Dropdown(["Aucun", "Occasionnel", "Frequent"], label="Fréquence de déplacement")

        predict_button = gr.Button("Lancer la prédiction")
        output = gr.Textbox(label="Résultat de la prédiction")

        predict_button.click(
            fn=process_input,
            inputs=[
                age, genre, revenu, statut_marital, departement, poste, niveau_hierarchique_poste,
                nombre_experiences_precedentes, annee_experience_totale, annees_dans_l_entreprise,
                annees_dans_le_poste_actuel, satisfaction_employee_environnement,
                note_evaluation_precedente, satisfaction_employee_nature_travail,
                satisfaction_employee_equipe, satisfaction_employee_equilibre_pro_perso,
                note_evaluation_actuelle, heure_supplementaires, augmentation_salaire_precedente_pourcent,
                nombre_participation_pee, nb_formations_suivies, distance_domicile_travail,
                niveau_education, domaine_etude, frequence_deplacement,
                annees_depuis_la_derniere_promotion, annes_sous_responsable_actuel
            ],
            outputs=output
        )

    return demo