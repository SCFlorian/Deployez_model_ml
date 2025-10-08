# Étape 1 : Image de base
FROM python:3.10

# Étape 2 : Définir le dossier de travail
WORKDIR /home/user/app

# Étape 3 : Copier tout le contenu du repo dans le conteneur
COPY . .

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port utilisé par Gradio
EXPOSE 7860

# Étape 6 : Lancer l'application
CMD ["python", "app.py"]