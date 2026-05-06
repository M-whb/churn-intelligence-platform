# Churn Intelligence Platform

Système intelligent de prédiction du churn client et d'évaluation du risque de revenus.  
Projet Data Science M2 — EFREI Paris | Promotion 2026

---

## Présentation

Ce projet conçoit un MVP complet de plateforme IA de rétention client, couvrant :

- La préparation et l'analyse exploratoire des données (EDA)
- La modélisation supervisée multi-algorithmes (ML + Deep Learning)
- L'évaluation comparative rigoureuse des modèles
- L'interprétabilité via Feature Importance et SHAP
- Un dashboard décisionnel interactif (Streamlit)
- Une API REST d'inférence deployable (FastAPI)

**Tâche prédictive :** Classification binaire — prédiction du churn (0 = Non-churn, 1 = Churn)  
**Dataset :** [Customer Churn Prediction Business Dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset) — 10 000 clients × 32 variables

---

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

---

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

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### 1. Préparer les données

Placer `customer_churn.csv` dans `data/raw/`, puis :

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

---

## Modèles implémentés

| Modèle | Type | ROC-AUC | F1-Score | Recall |
|---|---|---|---|---|
| Régression Logistique | ML — Baseline | 0.751 | 0.306 | 0.672 |
| Random Forest | ML — Ensemble | **0.791** | 0.332 | 0.412 |
| XGBoost | ML — Boosting | 0.773 | 0.336 | 0.456 |
| MLP | Deep Learning | 0.755 | **0.346** | 0.628 |

**Modèle recommandé :** Random Forest — meilleur ROC-AUC et meilleure stabilité.  
Le déséquilibre des classes (10.2% de churn) est géré via `class_weight='balanced'` et `scale_pos_weight`.

---

## Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | État du service et modèles chargés |
| GET | `/model-info` | Performances et informations des modèles |
| POST | `/predict` | Prédiction du churn pour un client |

### Exemple de requête `/predict`

```json
{
  "tenure_months": 3,
  "age": 35,
  "monthly_logins": 2,
  "csat_score": 1.5,
  "payment_failures": 2,
  "monthly_fee": 50.0,
  "total_revenue": 150.0,
  "contract_type": "Monthly",
  "model_name": "random_forest"
}
```

### Exemple de réponse

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.6821,
  "risk_level": "HIGH",
  "revenue_at_risk": 34.11,
  "model_used": "random_forest",
  "recommendations": [
    "Contacter le client pour régulariser les échecs de paiement.",
    "Proposer un appel de suivi — satisfaction client critique."
  ]
}
```

---

## Stack technique

| Catégorie | Technologie |
|---|---|
| Langage | Python 3.13 |
| ML | scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Interprétabilité | SHAP |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| API | FastAPI, Uvicorn, Pydantic |
| Sérialisation | Joblib |

---

## Auteurs

Projet réalisé dans le cadre du cours Data Science M2 — EFREI Paris  
Enseignante : Sarah Malaeb | Année : 2025-2026