# Churn Intelligence Platform


## Présentation

Ce projet conçoit un MVP complet de plateforme IA de rétention client, couvrant :

- La préparation et l'analyse exploratoire des données (EDA)
- La modélisation supervisée multi-algorithmes (ML + Deep Learning)
- L'évaluation comparative rigoureuse des modèles
- L'interprétabilité via Feature Importance et SHAP
- Un dashboard décisionnel interactif (Streamlit)
- Une API REST d'inférence deployable (FastAPI)

**Tâche prédictive :** Classification binaire — prédiction du churn (0 = Non-churn, 1 = Churn)  
**Dataset :** https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset


## Structure du projet

churn_project/
├── data/
│   ├── raw/                  # Dataset brut (customer_churn.csv)
│   └── processed/            # Données préparées + scaler.pkl
├── notebooks/
│   ├── 01_eda.ipynb          # Analyse exploratoire + preprocessing
│   └── 02_models.ipynb       # Modélisation + évaluation + SHAP
├── src/
│   ├── init.py
│   └── preprocessing.py      # Pipeline de preprocessing modulaire
├── models/                   # Modèles entraînés sauvegardés (.pkl, .keras)
├── dashboard/
│   └── app.py                # Dashboard Streamlit
├── api/
│   └── main.py               # API REST FastAPI
├── requirements.txt
└── README.md


## Installation

### Prérequis

- Python 3.13+
- pip

### Mise en place de l'environnement

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/churn_project.git
cd churn_project

# Créer et activer l'environnement virtuel
python -m venv venv

venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```


## Utilisation

### 1. Préparer les données

```bash
python src/preprocessing.py
```

Génère automatiquement `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` et `scaler.pkl` dans `data/processed/`.

### 2. Entraîner les modèles

Exécuter les notebooks dans l'ordre :

```bash
# Depuis VS Code ou Jupyter
notebooks/01_eda.ipynb      # EDA + preprocessing
notebooks/02_models.ipynb   # Entraînement + évaluation + SHAP
```

Les modèles sont sauvegardés automatiquement dans `models/`.

### 3. Lancer le dashboard

```bash
cd dashboard
streamlit run app.py
```

Ouvre automatiquement sur `http://localhost:8501`

### 4. Lancer l'API REST

```bash
uvicorn api.main:app --reload --port 8000
```

Disponible sur `http://localhost:8000`  
Documentation Swagger interactive : `http://localhost:8000/docs`



**Modèle recommandé :** Random Forest — meilleur ROC-AUC et meilleure stabilité.  


## Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | État du service et modèles chargés |
| GET | `/model-info` | Performances et informations des modèles |
| POST | `/predict` | Prédiction du churn pour un client |


## Auteurs

Angélique et Marie Wahba
