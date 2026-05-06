"""
preprocessing.py
================
Module de préparation des données pour le projet Churn Prediction.
Contient toutes les étapes de nettoyage, feature engineering,
encodage et split train/test.

Usage :
    from src.preprocessing import run_preprocessing
    X_train, X_test, y_train, y_test, scaler = run_preprocessing()
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONSTANTES
# ============================================================
COLS_TO_DROP   = ['customer_id', 'city', 'country']
TARGET         = 'churn'
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
CAT_COLS       = [
    'gender', 'customer_segment', 'signup_channel',
    'contract_type', 'payment_method', 'discount_applied',
    'price_increase_last_3m', 'complaint_type', 'survey_response'
]


# ============================================================
# ÉTAPES DU PIPELINE
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Charge le dataset brut depuis un fichier CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset introuvable : {filepath}")
    df = pd.read_csv(filepath)
    print(f"[load_data] Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage des données :
    - Suppression des colonnes non prédictives
    - Imputation des valeurs manquantes
    """
    df = df.copy()

    # Suppression des colonnes inutiles
    cols_existing = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_existing)
    print(f"[clean_data] Colonnes supprimées : {cols_existing}")

    # Imputation complaint_type : pas de plainte = 'No_complaint'
    if 'complaint_type' in df.columns:
        n_missing = df['complaint_type'].isnull().sum()
        df['complaint_type'] = df['complaint_type'].fillna('No_complaint')
        print(f"[clean_data] complaint_type : {n_missing} valeurs manquantes imputées")

    # Vérification finale
    remaining = df.isnull().sum().sum()
    print(f"[clean_data] Valeurs manquantes restantes : {remaining}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Création de nouvelles features métier :
    - login_per_month  : engagement normalisé par ancienneté
    - payment_risk     : exposition financière aux échecs de paiement
    - recency_risk     : risque lié à l'inactivité récente
    """
    df = df.copy()

    df['login_per_month'] = (
        df['monthly_logins'] / (df['tenure_months'] + 1)
    )
    df['payment_risk'] = (
        df['payment_failures'] * df['monthly_fee']
    )
    df['recency_risk'] = (
        df['last_login_days_ago'] / (df['avg_session_time'] + 1)
    )

    print("[feature_engineering] 3 features créées : "
          "login_per_month, payment_risk, recency_risk")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encoding des variables catégorielles.
    drop_first=True évite la multicolinéarité (dummy trap).
    """
    cat_existing = [c for c in CAT_COLS if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cat_existing, drop_first=True)
    print(f"[encode_categoricals] {len(cat_existing)} variables encodées — "
          f"dimensions : {df_encoded.shape}")
    return df_encoded


def split_and_scale(
    df: pd.DataFrame,
    output_dir: str = None
) -> tuple:
    """
    Split stratifié train/test + normalisation StandardScaler.

    Le scaler est fitté UNIQUEMENT sur le train set pour éviter
    tout data leakage vers le test set.

    Paramètres
    ----------
    df         : DataFrame encodé avec la colonne 'churn'
    output_dir : si renseigné, sauvegarde les fichiers CSV + scaler.pkl

    Retourne
    --------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split stratifié — préserve la proportion de churn dans train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[split_and_scale] Train : {X_train.shape[0]} lignes "
          f"({y_train.mean()*100:.1f}% churn)")
    print(f"[split_and_scale] Test  : {X_test.shape[0]} lignes "
          f"({y_test.mean()*100:.1f}% churn)")

    # Normalisation — fit sur train uniquement
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    print("[split_and_scale] StandardScaler appliqué (fit sur train uniquement)")

    # Sauvegarde optionnelle
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        X_train_scaled.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test_scaled.to_csv(f"{output_dir}/X_test.csv",   index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv",        index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv",          index=False)
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
        print(f"[split_and_scale] Fichiers sauvegardés dans : {output_dir}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================
# PIPELINE COMPLET
# ============================================================

def run_preprocessing(
    raw_path:   str = None,
    output_dir: str = None
) -> tuple:
    """
    Exécute le pipeline complet de preprocessing :
    load → clean → feature_engineering → encode → split_and_scale

    Paramètres
    ----------
    raw_path   : chemin vers customer_churn.csv
                 (défaut : data/raw/customer_churn.csv relatif à ce fichier)
    output_dir : dossier de sauvegarde des fichiers préparés
                 (défaut : data/processed relatif à ce fichier)

    Retourne
    --------
    X_train, X_test, y_train, y_test, scaler
    """
    # Chemins par défaut
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raw_path is None:
        raw_path = os.path.join(base, 'data', 'raw', 'customer_churn.csv')
    if output_dir is None:
        output_dir = os.path.join(base, 'data', 'processed')

    print("=" * 50)
    print("  PIPELINE PREPROCESSING — CHURN PREDICTION")
    print("=" * 50)

    df = load_data(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        df, output_dir=output_dir
    )

    print("=" * 50)
    print("  PREPROCESSING TERMINÉ")
    print(f"  Features : {X_train.shape[1]} | "
          f"Train : {len(X_train)} | Test : {len(X_test)}")
    print("=" * 50)

    return X_train, X_test, y_train, y_test, scaler


# ============================================================
# EXÉCUTION DIRECTE
# ============================================================
if __name__ == "__main__":
    run_preprocessing()