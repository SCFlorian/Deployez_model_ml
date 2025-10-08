---
title: Deployez Modele ML
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Déploiement d’un modèle de machine learning

## Description
Ce projet reprend le modèle construit dans le projet *« Classifiez automatiquement des informations »* et en prépare le déploiement.  
L’objectif est de rendre le modèle accessible via une API développée avec **FastAPI**, d’intégrer un suivi Git complet basé sur **Gitflow** et de gérer la persistance des données dans une base **PostgreSQL**.

---

## Organisation Gitflow

Le cycle de vie du projet suit strictement le workflow **Gitflow** :  

1. **Feature branches** : développement d’une nouvelle fonctionnalité.  
   - `feature/feature-engineering` : traitement des données brutes pour générer des données nettoyées.  
   - `feature/feature-prediction` : utilisation des données nettoyées pour générer des prédictions.  

2. **Develop** : intègre les fonctionnalités terminées.  

3. **Release** : stabilisation avant passage en production. Un **tag de version** est créé (ex : `v1.0.0`).  

4. **Main** : branche de production, déployée automatiquement sur Hugging Face Spaces.  

**Exemple de cycle :**  
- Création de `feature/feature-engineering` → merge dans `develop`
- Merge de `develop` → `release/v1.0.0` avec création du tag
- Merge de `release/v1.0.0` → `main`
- Même logique ensuite pour `feature/feature-prediction`

---

## Installation

### 1. Cloner le dépôt
```bash
git clone git@github.com:SCFlorian/deployez_modele_ml.git
cd deployez_modele_ml
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate    # Mac / Linux
.venv\Scripts\activate       # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Organisation du projet
- models/ : modèle sauvegardé (model.pkl)
- src/ : scripts Python
- api/ : API FastAPI
- tests/ : tests unitaires
- app.py : script principal pour Hugging Face Spaces

## Utilisation
L’API permet :
- de saisir des données via un formulaire
- de stocker ces données en BDD
- d’appliquer automatiquement les transformations et le feature engineering
- de renvoyer une prédiction persistée avec ses métadonnées

## Authentification & Sécurisation
- Authentification par token
- Gestion des secrets via .env

## Historique

## Gestion des environnements
- Développement : branches feature/*
- Test : exécution automatique des tests unitaires via GitHub Actions à chaque push
- Release : stabilisation et tagging (ex. v1.0.0, v1.1.0)
- Production : branche main, déployée automatiquement sur Hugging Face Spaces
- Fusion vers main uniquement via Pull Request validée