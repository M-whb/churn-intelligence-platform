from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# INITIALISATION
# ============================================================
app = FastAPI(
    title="Churn Prediction API",
    description="API REST de prédiction du churn client — EFREI M2 Data Science",
    version="1.0.0"
)

# Chargement des modèles au démarrage
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    models = {
        "logistic_regression": joblib.load(f"{BASE}/models/logistic_regression.pkl"),
        "random_forest":       joblib.load(f"{BASE}/models/random_forest.pkl"),
        "xgboost":             joblib.load(f"{BASE}/models/xgboost.pkl"),
    }
    scaler       = joblib.load(f"{BASE}/data/processed/scaler.pkl")
    feature_cols = pd.read_csv(f"{BASE}/data/processed/X_train.csv", nrows=0).columns.tolist()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR   = str(e)

# ============================================================
# SCHÉMA D'ENTRÉE — toutes les features du client
# ============================================================
class ClientFeatures(BaseModel):
    # Profil
    tenure_months:      int   = Field(..., ge=1, le=120,
                                      description="Ancienneté en mois (1-120)")
    age:                int   = Field(..., ge=18, le=100,
                                      description="Âge du client")
    # Comportement
    monthly_logins:     int   = Field(..., ge=0, le=100,
                                      description="Connexions mensuelles")
    weekly_active_days: int   = Field(..., ge=0, le=7,
                                      description="Jours actifs par semaine")
    avg_session_time:   float = Field(..., ge=0.0,
                                      description="Durée moyenne des sessions (min)")
    features_used:      int   = Field(..., ge=0,
                                      description="Nombre de fonctionnalités utilisées")
    usage_growth_rate:  float = Field(0.0,
                                      description="Taux de croissance d'usage")
    last_login_days_ago:int   = Field(..., ge=0,
                                      description="Jours depuis la dernière connexion")
    # Financier
    monthly_fee:        float = Field(..., ge=0,
                                      description="Frais mensuels (€)")
    total_revenue:      float = Field(..., ge=0,
                                      description="Revenu total généré (€)")
    payment_failures:   int   = Field(..., ge=0, le=20,
                                      description="Nombre d'échecs de paiement")
    # Support
    support_tickets:    int   = Field(..., ge=0,
                                      description="Tickets support ouverts")
    avg_resolution_time:float = Field(0.0, ge=0,
                                      description="Temps moyen de résolution (h)")
    csat_score:         float = Field(..., ge=1.0, le=5.0,
                                      description="Score de satisfaction (1-5)")
    escalations:        int   = Field(0, ge=0,
                                      description="Nombre d'escalades")
    # Marketing
    nps_score:          int   = Field(..., ge=-100, le=100,
                                      description="Net Promoter Score (-100 à 100)")
    email_open_rate:    float = Field(0.5, ge=0.0, le=1.0,
                                      description="Taux d'ouverture des emails (0-1)")
    marketing_click_rate: float = Field(0.25, ge=0.0, le=1.0,
                                      description="Taux de clic marketing (0-1)")
    referral_count:     int   = Field(0, ge=0,
                                      description="Nombre de références")
    # Catégorielles
    gender:             str   = Field("Male",
                                      description="Male ou Female")
    customer_segment:   str   = Field("Individual",
                                      description="Individual, SME ou Enterprise")
    signup_channel:     str   = Field("Web",
                                      description="Web, Mobile ou Partner")
    contract_type:      str   = Field("Monthly",
                                      description="Monthly, Yearly ou Two-Year")
    payment_method:     str   = Field("Card",
                                      description="Card, PayPal ou BankTransfer")
    discount_applied:   str   = Field("No",
                                      description="Yes ou No")
    price_increase_last_3m: str = Field("No",
                                      description="Yes ou No")
    complaint_type:     str   = Field("No_complaint",
                                      description="Billing, Technical, Service ou No_complaint")
    survey_response:    str   = Field("Neutral",
                                      description="Satisfied, Neutral ou Unsatisfied")
    # Modèle à utiliser
    model_name:         str   = Field("random_forest",
                                      description="logistic_regression, random_forest ou xgboost")

    @field_validator('model_name')
    @classmethod
    def validate_model(cls, v):
        allowed = ["logistic_regression", "random_forest", "xgboost"]
        if v not in allowed:
            raise ValueError(f"model_name doit être parmi : {allowed}")
        return v


# ============================================================
# FONCTION DE PRÉPARATION DES FEATURES
# ============================================================
def prepare_features(client: ClientFeatures) -> pd.DataFrame:
    """Transforme les données client en vecteur de features aligné sur X_train."""

    # Création du DataFrame de base
    row = pd.DataFrame([{
        'age':                    client.age,
        'tenure_months':          client.tenure_months,
        'monthly_logins':         client.monthly_logins,
        'weekly_active_days':     client.weekly_active_days,
        'avg_session_time':       client.avg_session_time,
        'features_used':          client.features_used,
        'usage_growth_rate':      client.usage_growth_rate,
        'last_login_days_ago':    client.last_login_days_ago,
        'monthly_fee':            client.monthly_fee,
        'total_revenue':          client.total_revenue,
        'payment_failures':       client.payment_failures,
        'support_tickets':        client.support_tickets,
        'avg_resolution_time':    client.avg_resolution_time,
        'csat_score':             client.csat_score,
        'escalations':            client.escalations,
        'nps_score':              client.nps_score,
        'email_open_rate':        client.email_open_rate,
        'marketing_click_rate':   client.marketing_click_rate,
        'referral_count':         client.referral_count,
        # Features engineerées
        'login_per_month':        client.monthly_logins / (client.tenure_months + 1),
        'payment_risk':           client.payment_failures * client.monthly_fee,
        'recency_risk':           client.last_login_days_ago / (client.avg_session_time + 1),
    }])

    # One-Hot Encoding manuel — on doit reproduire exactement l'encodage du preprocessing
    cat_dummies = {
        'gender_Male':                    int(client.gender == 'Male'),
        'customer_segment_Individual':    int(client.customer_segment == 'Individual'),
        'customer_segment_SME':           int(client.customer_segment == 'SME'),
        'signup_channel_Mobile':          int(client.signup_channel == 'Mobile'),
        'signup_channel_Web':             int(client.signup_channel == 'Web'),
        'contract_type_Monthly':          int(client.contract_type == 'Monthly'),
        'contract_type_Yearly':           int(client.contract_type == 'Yearly'),
        'payment_method_Card':            int(client.payment_method == 'Card'),
        'payment_method_PayPal':          int(client.payment_method == 'PayPal'),
        'discount_applied_Yes':           int(client.discount_applied == 'Yes'),
        'price_increase_last_3m_Yes':     int(client.price_increase_last_3m == 'Yes'),
        'complaint_type_No_complaint':    int(client.complaint_type == 'No_complaint'),
        'complaint_type_Service':         int(client.complaint_type == 'Service'),
        'complaint_type_Technical':       int(client.complaint_type == 'Technical'),
        'survey_response_Satisfied':      int(client.survey_response == 'Satisfied'),
        'survey_response_Unsatisfied':    int(client.survey_response == 'Unsatisfied'),
    }
    for col, val in cat_dummies.items():
        row[col] = val

    # Aligner sur les colonnes exactes de X_train (ordre + colonnes manquantes)
    row = row.reindex(columns=feature_cols, fill_value=0)

    # Application du scaler
    row_scaled = scaler.transform(row)
    return pd.DataFrame(row_scaled, columns=feature_cols)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Vérifie que le service est actif et les modèles chargés."""
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"Modèles non chargés : {LOAD_ERROR}"
        )
    return {
        "status":          "healthy",
        "models_loaded":   list(models.keys()),
        "n_features":      len(feature_cols),
        "api_version":     "1.0.0"
    }


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    """Informations sur les modèles disponibles et leurs performances."""
    return {
        "models_available": list(models.keys()),
        "default_model":    "random_forest",
        "performances": {
            "logistic_regression": {"roc_auc": 0.751, "f1": 0.306, "recall": 0.672},
            "random_forest":       {"roc_auc": 0.791, "f1": 0.332, "recall": 0.412},
            "xgboost":             {"roc_auc": 0.773, "f1": 0.336, "recall": 0.456},
        },
        "target":         "churn (0=No, 1=Yes)",
        "n_features":     len(feature_cols),
        "training_size":  8000,
        "test_size":      2000
    }


@app.post("/predict", tags=["Prédiction"])
def predict(client: ClientFeatures):
    """
    Prédit la probabilité de churn pour un client.

    Retourne :
    - **churn_prediction** : 0 (non-churn) ou 1 (churn)
    - **churn_probability** : probabilité entre 0 et 1
    - **risk_level** : LOW / MEDIUM / HIGH
    - **revenue_at_risk** : estimation du revenu mensuel à risque (€)
    - **recommendations** : liste d'actions suggérées
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modèles non disponibles")

    try:
        X = prepare_features(client)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur de préparation des features : {str(e)}")

    model = models[client.model_name]
    proba = float(model.predict_proba(X)[0][1])
    pred  = int(proba >= 0.5)

    # Niveau de risque
    if proba >= 0.5:
        risk_level = "HIGH"
    elif proba >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Revenu à risque
    revenue_at_risk = round(client.monthly_fee * proba, 2)

    # Recommandations automatiques
    recommendations = []
    if client.payment_failures > 0:
        recommendations.append("Contacter le client pour régulariser les échecs de paiement.")
    if client.csat_score < 3.0:
        recommendations.append("Proposer un appel de suivi — satisfaction client critique.")
    if client.tenure_months < 6:
        recommendations.append("Renforcer l'onboarding — client récent à risque élevé.")
    if client.monthly_logins < 5:
        recommendations.append("Lancer une campagne de réengagement personnalisée.")
    if client.last_login_days_ago > 14:
        recommendations.append("Relancer avec une offre exclusive — inactivité détectée.")
    if not recommendations:
        recommendations.append("Profil stable — maintenir la relation client standard.")

    return {
        "churn_prediction":  pred,
        "churn_probability": round(proba, 4),
        "risk_level":        risk_level,
        "revenue_at_risk":   revenue_at_risk,
        "model_used":        client.model_name,
        "recommendations":   recommendations
    }