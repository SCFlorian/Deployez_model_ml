import gradio as gr
import pandas as pd
import joblib
import requests

# FONCTIONS DE TEST

def test_feature_engineering():
    """Test de la fonction de feature engineering seule."""
    return "Endpoint 'Feature Engineering' : Fonction disponible et opérationnelle."

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


# CONFIG
API_URL = "/predict"

# PROCESS DE PRÉDICTION

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
        return f"{data['message']} — probabilité : {data['probability']:.3f}"
    except Exception as e:
        return f"Erreur : {e}"


# INTERFACE GRADIO
def build_interface():
    gr.close_all()

    css_style = """
    * { font-family: 'Inter', sans-serif !important; }
    body { background-color: #f8f9fb !important; }
    h1, h2, h3 { color: #2b2d42 !important; font-weight: 700 !important; }
    label { color: #1a1b1f !important; font-weight: 500; }
    select {
        appearance: auto !important;
        -webkit-appearance: menulist !important;
        font-size: 14px !important;
        height: 2.3em !important;
        padding: 0.3em !important;
        border-radius: 6px !important;
    }
    button {
        font-weight: 600 !important;
        border-radius: 8px !important;
        height: 2.6em !important;
        background-color: #4f46e5 !important;
        color: white !important;
    }
    button:hover {
        background-color: #4338ca !important;
    }
    .gradio-container {
        max-width: 1100px !important;
        margin: auto !important;
    }
    """

    with gr.Blocks(
        title="Déployez votre Modèle RH - Prédiction de Départ",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet"),
        css=css_style,
    ) as demo:

        # ====== EN-TÊTE ======
        gr.Markdown(
            """
            <div style='text-align: center; margin-bottom: 30px;'>
                <h1> Déployez votre modèle de <span style='color:#4f46e5;'>Machine Learning RH</span></h1>
                <p style='font-size:16px;color:#555;'>Prédisez le risque de départ d’un employé à partir de ses caractéristiques.</p>
            </div>
            """
        )

        # ====== TEST DES ENDPOINTS ======
        gr.Markdown("## Test des endpoints disponibles")

        with gr.Row():
            with gr.Column():
                test_output1 = gr.Textbox(label="Résultat Feature Engineering")
                btn1 = gr.Button("Tester Feature Engineering")
                btn1.click(fn=test_feature_engineering, outputs=test_output1)

            with gr.Column():
                test_output2 = gr.Textbox(label="Résultat Scaling")
                btn2 = gr.Button("Tester Scaling")
                btn2.click(fn=test_scaling, outputs=test_output2)

            with gr.Column():
                test_output3 = gr.Textbox(label="Résultat Modèle")
                btn3 = gr.Button("Tester Modèle")
                btn3.click(fn=test_model, outputs=test_output3)

        gr.Markdown("---")

        # ====== FORMULAIRE ======
        gr.Markdown("## Saisir les informations de l’employé")

        with gr.Row():
            age = gr.Number(label="Âge", minimum=18, maximum=60, step=1)
            genre = gr.Radio(choices=["M", "F"], label="Genre")
            revenu = gr.Number(label="Revenu mensuel (€)", minimum=1000, maximum=20000, step=100)

        with gr.Row():
            statut_marital = gr.Dropdown(["Célibataire", "Marié", "Divorcé"], label="Statut marital")
            departement = gr.Dropdown(["Commercial", "Consulting", "Ressources Humaines"], label="Département")
            poste = gr.Dropdown(
                [
                    "Assistant de Direction", "Cadre Commercial", "Consultant",
                    "Directeur Technique", "Manager", "Représentant Commercial",
                    "Ressources Humaines", "Senior Manager", "Tech Lead"
                ],
                label="Poste"
            )

        with gr.Row():
            with gr.Column():
                niveau_hierarchique_poste = gr.Number(label="Niveau hiérarchique", minimum=1, maximum=5)
                nombre_experiences_precedentes = gr.Number(label="Expériences précédentes", minimum=1, maximum=9)
                annee_experience_totale = gr.Number(label="Années d'expérience totale", minimum=0, maximum=40)
                annees_dans_l_entreprise = gr.Number(label="Années dans l’entreprise", minimum=0, maximum=40)
                annees_dans_le_poste_actuel = gr.Number(label="Années dans le poste actuel", minimum=0, maximum=18)
                annees_depuis_la_derniere_promotion = gr.Number(label="Années depuis la dernière promotion", minimum=0, maximum=15)
                annes_sous_responsable_actuel = gr.Number(label="Années sous responsable actuel", minimum=0, maximum=17)

            with gr.Column():
                satisfaction_employee_environnement = gr.Slider(1, 4, step=1, label="Satisfaction environnement")
                satisfaction_employee_nature_travail = gr.Slider(1, 4, step=1, label="Satisfaction nature du travail")
                satisfaction_employee_equipe = gr.Slider(1, 4, step=1, label="Satisfaction équipe")
                satisfaction_employee_equilibre_pro_perso = gr.Slider(1, 4, step=1, label="Équilibre pro/perso")
                note_evaluation_precedente = gr.Slider(1, 4, step=1, label="Évaluation précédente")
                note_evaluation_actuelle = gr.Slider(3, 4, step=1, label="Évaluation actuelle")

        with gr.Row():
            heure_supplementaires = gr.Radio(["Oui", "Non"], label="Heures supplémentaires")
            augmentation_salaire_precedente_pourcent = gr.Dropdown(
                [f"{i/100:.2f}" for i in range(11, 26)], label="Augmentation du salaire précédente (%)"
            )
            nombre_participation_pee = gr.Number(label="Participations PEE", minimum=0, maximum=3)
            nb_formations_suivies = gr.Number(label="Formations suivies", minimum=0, maximum=6)
            distance_domicile_travail = gr.Number(label="Distance domicile-travail (km)", minimum=1, maximum=29)

        with gr.Row():
            niveau_education = gr.Slider(1, 5, step=1, label="Niveau d'éducation")
            domaine_etude = gr.Dropdown(
                ["Autre", "Entrepreneuriat", "InfraCloud", "Marketing", "Ressources Humaines", "Transformation Digitale"],
                label="Domaine d’étude"
            )
            frequence_deplacement = gr.Dropdown(["Aucun", "Occasionnel", "Fréquent"], label="Fréquence de déplacement")

        gr.Markdown("---")
        predict_button = gr.Button("Lancer la prédiction")
        output = gr.Textbox(label="Résultat de la prédiction", lines=3)

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