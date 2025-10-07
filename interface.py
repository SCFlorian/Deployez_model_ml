import gradio as gr
import joblib
import requests

# --- Tests simples ---
def test_feature_engineering():
    return "Endpoint 'Feature Engineering' : opérationnel."

def test_scaling():
    try:
        joblib.load("models/standard_scaler.pkl")
        return "Endpoint 'Scaling' : scaler chargé avec succès."
    except Exception as e:
        return f"Endpoint 'Scaling' : Erreur — {e}"

def test_model():
    try:
        joblib.load("models/final_model.pkl")
        with open("models/threshold.txt", "r") as f:
            threshold = float(f.read())
        return f"Endpoint 'Model' : modèle chargé (seuil {threshold:.3f})"
    except Exception as e:
        return f"Endpoint 'Model' : Erreur — {e}"

# --- Prédiction via API ---
API_URL = "/predict"

def process_input(**kwargs):
    try:
        r = requests.post(API_URL, json=kwargs, timeout=20)
        r.raise_for_status()
        data = r.json()
        return f"{data.get('message', '')} — probabilité : {data.get('probability', 0):.3f}"
    except Exception as e:
        return f"Erreur d’appel API : {e}"

# --- Interface Gradio ---
def build_interface():
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="gray"
    ).set(
        body_background_fill="#f8f9fb",
        background_fill_primary="#ffffff",
        border_radius="10px"
    )

    css = """
    .gradio-container {max-width: 1100px !important; margin: auto !important;}
    h1, h2, h3 {color: #1f2937 !important; font-weight: 700 !important;}
    .gr-box {border-radius: 12px !important; box-shadow: 0 3px 10px rgba(0,0,0,0.08) !important;}
    button {font-weight: 600 !important;}
    """

    with gr.Blocks(theme=theme, css=css, title="Déployez votre modèle RH") as demo:
        gr.Markdown(
            """
            # Déployez votre modèle de Machine Learning RH  
            *Prédisez le risque de départ d’un employé à partir de ses caractéristiques professionnelles.*
            """
        )

        # ---- Bloc tests ----
        with gr.Group():
            gr.Markdown("## Vérification des endpoints")

            with gr.Row():
                with gr.Column():
                    test_output1 = gr.Textbox(label="Résultat Feature Engineering", interactive=False)
                    gr.Button("Tester Feature Engineering", variant="primary").click(
                        fn=test_feature_engineering, outputs=test_output1
                    )

                with gr.Column():
                    test_output2 = gr.Textbox(label="Résultat Scaling", interactive=False)
                    gr.Button("Tester Scaling", variant="primary").click(
                        fn=test_scaling, outputs=test_output2
                    )

                with gr.Column():
                    test_output3 = gr.Textbox(label="Résultat Modèle", interactive=False)
                    gr.Button("Tester Modèle", variant="primary").click(
                        fn=test_model, outputs=test_output3
                    )

        # ---- Bloc formulaire ----
        gr.Markdown("---")
        gr.Markdown("## Saisir les informations de l’employé")

        with gr.Group():
            with gr.Row():
                age = gr.Number(label="Âge", value=35, minimum=18, maximum=60)
                genre = gr.Radio(["M", "F"], label="Genre", value="M")
                revenu = gr.Number(label="Revenu mensuel (€)", value=5000)

            with gr.Row():
                statut_marital = gr.Dropdown(
                    ["Célibataire", "Marié", "Divorcé"], label="Statut marital", value="Célibataire"
                )
                departement = gr.Dropdown(
                    ["Commercial", "Consulting", "Ressources Humaines"],
                    label="Département", value="Commercial"
                )
                poste = gr.Dropdown(
                    [
                        "Assistant de Direction", "Cadre Commercial", "Consultant", "Directeur Technique",
                        "Manager", "Représentant Commercial", "Ressources Humaines", "Senior Manager", "Tech Lead"
                    ],
                    label="Poste", value="Consultant"
                )

            with gr.Row():
                with gr.Column():
                    niveau_hierarchique_poste = gr.Slider(1, 5, value=2, label="Niveau hiérarchique")
                    nombre_experiences_precedentes = gr.Number(value=1, label="Expériences précédentes")
                    annee_experience_totale = gr.Number(value=8, label="Années d'expérience totale")
                    annees_dans_l_entreprise = gr.Number(value=3, label="Années dans l’entreprise")
                    annees_dans_le_poste_actuel = gr.Number(value=2, label="Années dans le poste actuel")

                with gr.Column():
                    satisfaction_employee_environnement = gr.Slider(1, 4, value=3, step=1, label="Satisfaction environnement")
                    satisfaction_employee_nature_travail = gr.Slider(1, 4, value=3, step=1, label="Satisfaction nature du travail")
                    satisfaction_employee_equipe = gr.Slider(1, 4, value=3, step=1, label="Satisfaction équipe")
                    satisfaction_employee_equilibre_pro_perso = gr.Slider(1, 4, value=3, step=1, label="Équilibre pro/perso")

            with gr.Row():
                note_evaluation_precedente = gr.Slider(1, 4, value=3, label="Note évaluation précédente")
                note_evaluation_actuelle = gr.Slider(1, 4, value=3, label="Note évaluation actuelle")
                niveau_education = gr.Slider(1, 5, value=3, label="Niveau d’éducation")

            with gr.Row():
                heure_supplementaires = gr.Radio(["Oui", "Non"], label="Heures supplémentaires", value="Non")
                augmentation_salaire_precedente_pourcent = gr.Dropdown(
                    [f"{i/100:.2f}" for i in range(11, 26)],
                    label="Augmentation de salaire précédente (%)", value="0.15"
                )
                nombre_participation_pee = gr.Number(value=1, label="Participations PEE")

            with gr.Row():
                nb_formations_suivies = gr.Number(value=2, label="Formations suivies")
                distance_domicile_travail = gr.Number(value=10, label="Distance domicile-travail (km)")
                domaine_etude = gr.Dropdown(
                    ["Autre", "Entrepreneuriat", "InfraCloud", "Marketing", "RH", "Transformation Digitale"],
                    label="Domaine d’étude", value="Autre"
                )

            with gr.Row():
                frequence_deplacement = gr.Dropdown(
                    ["Aucun", "Occasionnel", "Fréquent"], label="Fréquence de déplacement", value="Occasionnel"
                )
                annees_depuis_la_derniere_promotion = gr.Number(value=1, label="Années depuis la dernière promotion")
                annes_sous_responsable_actuel = gr.Number(value=2, label="Années sous le responsable actuel")

        # ---- Bouton de prédiction ----
        predict_button = gr.Button("🚀 Lancer la prédiction", variant="primary", size="lg")
        output = gr.Textbox(label="Résultat de la prédiction", lines=2, interactive=False)

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