# Déploiement d'un modèle de Machine Learning - Explication CI/CD

## Conventions de code
- Respecter le style Python PEP8.
- Bien commenter le code et ajouter des docstrings simples pour les fonctions.

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

## CI/CD
- Les actions GitHub se trouvent dans .github/workflows/ci.yaml
- Le pipeline s’exécute automatiquement quand :
    - On fait un push sur main,
    - Ou une pull request vers main.

## Étapes du pipeline :
	1.	Installer Python et les dépendances.
	2.	Lancer les tests (dans tests/).
	3.	Déployer sur Hugging Face Spaces si les tests passent.

## Tests
- Les tests se trouvent dans tests/.
- Pour lancer les tests en local :
```bash
    pytest tests/
```