import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')


# CONFIG

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = r"C:\Users\wahba\churn_project"

# CHARGEMENT (mis en cache)

@st.cache_resource
def load_models():
    return {
        'Régression Logistique': joblib.load(f'{BASE}/models/logistic_regression.pkl'),
        'Random Forest':         joblib.load(f'{BASE}/models/random_forest.pkl'),
        'XGBoost':               joblib.load(f'{BASE}/models/xgboost.pkl'),
    }

@st.cache_resource
def load_scaler():
    return joblib.load(f'{BASE}/data/processed/scaler.pkl')

def load_data():
    X_test  = pd.read_csv(f'{BASE}/data/processed/X_test.csv')
    y_test  = pd.read_csv(f'{BASE}/data/processed/y_test.csv').iloc[:, 0]
    X_train = pd.read_csv(f'{BASE}/data/processed/X_train.csv')
    y_train = pd.read_csv(f'{BASE}/data/processed/y_train.csv').iloc[:, 0]
    df_raw  = pd.read_csv(f'{BASE}/data/raw/customer_churn.csv')
    return X_test, y_test, X_train, y_train, df_raw

models  = load_models()
scaler  = load_scaler()
X_test, y_test, X_train, y_train, df_raw = load_data()
FEATURE_COLS = X_train.columns.tolist()


# SIDEBAR

st.sidebar.title("📊 Churn Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Vue d'ensemble",
     "📈 Performance modèles",
     "🔍 Interprétabilité",
     "🎯 Prédiction client"]
)
st.sidebar.markdown("---")
model_choice  = st.sidebar.selectbox(
    "Modèle actif",
    list(models.keys()),
    index=1
)
active_model = models[model_choice]


# Préparer un vecteur de features à partir des saisies

def build_feature_vector(
    tenure_months, monthly_logins, csat_score, nps_score,
    monthly_fee, total_revenue, payment_failures,
    weekly_active_days, avg_session_time, support_tickets,
    last_login_days, gender, customer_segment, signup_channel,
    contract_type, payment_method, discount_applied,
    price_increase_last_3m, complaint_type, survey_response
):
    """Construit et scale un vecteur de features aligné sur X_train."""

    # Vecteur à zéro aligné sur les colonnes d'entraînement
    row = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLS)

    # Variables numériques 
    num = {
        'tenure_months':        tenure_months,
        'monthly_logins':       monthly_logins,
        'csat_score':           csat_score,
        'nps_score':            nps_score,
        'monthly_fee':          monthly_fee,
        'total_revenue':        total_revenue,
        'payment_failures':     payment_failures,
        'weekly_active_days':   weekly_active_days,
        'avg_session_time':     avg_session_time,
        'support_tickets':      support_tickets,
        'last_login_days_ago':  last_login_days,
        # features engineerées
        'login_per_month':      monthly_logins / (tenure_months + 1),
        'payment_risk':         payment_failures * monthly_fee,
        'recency_risk':         last_login_days / (avg_session_time + 1),
        # valeurs par défaut pour colonnes non saisies
        'age':                  35,
        'features_used':        3,
        'usage_growth_rate':    0.0,
        'avg_resolution_time':  0.0,
        'escalations':          0,
        'email_open_rate':      0.5,
        'marketing_click_rate': 0.25,
        'referral_count':       0,
    }
    for col, val in num.items():
        if col in row.columns:
            row[col] = float(val)

    # Encodage OHE 
    # Pour chaque variable catégorielle, on active la colonne
    # correspondant à la modalité saisie (si elle existe dans X_train)
    ohe = {
        'gender':                 gender,
        'customer_segment':       customer_segment,
        'signup_channel':         signup_channel,
        'contract_type':          contract_type,
        'payment_method':         payment_method,
        'discount_applied':       discount_applied,
        'price_increase_last_3m': price_increase_last_3m,
        'complaint_type':         complaint_type,
        'survey_response':        survey_response,
    }
    for orig, val in ohe.items():
        col_name = f"{orig}_{val}"
        if col_name in row.columns:
            row[col_name] = 1.0

    # Normalisation 
    row_scaled = pd.DataFrame(
        scaler.transform(row),
        columns=FEATURE_COLS
    )
    return row_scaled


# VUE D'ENSEMBLE

if page == "🏠 Vue d'ensemble":
    st.title("Vue d'ensemble — Rétention Client")
    st.markdown("Plateforme décisionnelle de prédiction du churn et d'analyse du risque de revenus.")

    y_pred_all  = active_model.predict(X_test)
    y_proba_all = active_model.predict_proba(X_test)[:, 1]
    n_churn     = int(y_pred_all.sum())
    mean_fee    = df_raw['monthly_fee'].mean()
    revenue_risk = round(n_churn * mean_fee, 0)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients analysés",       f"{len(X_test):,}")
    c2.metric("Clients à risque",       f"{n_churn:,}",
              f"{n_churn/len(X_test)*100:.1f}% du portefeuille")
    c3.metric("Revenu mensuel à risque", f"{revenue_risk:,.0f} €")
    c4.metric("Modèle actif", model_choice.split()[0] + "...",
              "Changer dans la barre latérale")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribution du churn")
        churn_counts = df_raw['churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=['Non-Churn', 'Churn'],
            color_discrete_sequence=['#2196F3', '#E53935'],
            hole=0.4
        )
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Taux de churn par type de contrat")
        churn_by_contract = df_raw.groupby('contract_type')['churn'].mean() * 100
        fig2 = px.bar(
            x=churn_by_contract.index,
            y=churn_by_contract.values,
            color=churn_by_contract.values,
            color_continuous_scale=['#B5D4F4', '#E53935'],
            labels={'x': 'Type de contrat', 'y': 'Taux de churn (%)'}
        )
        fig2.update_layout(showlegend=False, height=300,
                           coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Distribution des probabilités prédites")
    fig3 = px.histogram(
        x=y_proba_all, nbins=50,
        color_discrete_sequence=['#7B1FA2'],
        labels={'x': 'Probabilité de churn', 'y': 'Nombre de clients'}
    )
    fig3.add_vline(x=0.5, line_dash='dash', line_color='red',
                   annotation_text='Seuil 0.5')
    fig3.update_layout(height=280, margin=dict(t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 10 clients à risque")
    risk_df = X_test.copy()
    risk_df['proba_churn'] = y_proba_all
    risk_df['prediction']  = y_pred_all
    top_risk = risk_df.nlargest(10, 'proba_churn')[
        ['proba_churn', 'csat_score', 'tenure_months',
         'monthly_logins', 'payment_failures']
    ].round(3)
    top_risk['proba_churn'] = top_risk['proba_churn'].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(top_risk, use_container_width=True)


# PERFORMANCE MODÈLES

elif page == "📈 Performance modèles":
    st.title("Comparaison des performances")

    perf_data = {
        'Modèle':    ['Régression Logistique', 'Random Forest', 'XGBoost', 'MLP'],
        'Accuracy':  [0.6895, 0.8310, 0.8160, 0.7575],
        'Precision': [0.1983, 0.2781, 0.2657, 0.2384],
        'Recall':    [0.6716, 0.4118, 0.4559, 0.6275],
        'F1-Score':  [0.3061, 0.3320, 0.3357, 0.3455],
        'ROC-AUC':   [0.7510, 0.7908, 0.7728, 0.7554],
    }
    df_perf = pd.DataFrame(perf_data).set_index('Modèle')

    st.subheader("Tableau comparatif")
    st.dataframe(
        df_perf.style
               .highlight_max(axis=0, color='#C8E6C9')
               .highlight_min(axis=0, color='#FFCDD2')
               .format("{:.4f}"),
        use_container_width=True
    )
    st.caption("Vert = meilleur score | Rouge = score le plus faible")

    st.subheader("Courbes ROC")
    fig_roc = go.Figure()
    colors  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for (name, model), color in zip(models.items(), colors):
        yp   = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yp)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"{name} (AUC={auc(fpr,tpr):.3f})",
            line=dict(color=color, width=2)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Baseline', line=dict(color='gray', dash='dash')
    ))
    fig_roc.update_layout(
        xaxis_title='Taux de faux positifs',
        yaxis_title='Recall (vrais positifs)',
        height=420, margin=dict(t=20)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader(f"Matrice de confusion — {model_choice}")
    y_pred = active_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Churn', 'Churn'],
                yticklabels=['Non-Churn', 'Churn'], ax=ax)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    st.pyplot(fig_cm)

    tn, fp, fn, tp = cm.ravel()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vrais Positifs (churners détectés)", tp)
    c2.metric("Faux Négatifs (churners ratés)", fn,
              delta=f"-{fn}", delta_color="inverse")
    c3.metric("Vrais Négatifs", tn)
    c4.metric("Faux Positifs", fp)



# INTERPRÉTABILITÉ

elif page == "🔍 Interprétabilité":
    st.title("Interprétabilité des modèles")

    st.subheader("Feature Importance — Random Forest")
    rf_model = models['Random Forest']
    feat_imp = pd.Series(
        rf_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(15)

    fig_fi = px.bar(
        x=feat_imp.values, y=feat_imp.index,
        orientation='h',
        color=feat_imp.values,
        color_continuous_scale=['#B5D4F4', '#0C447C'],
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    fig_fi.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500, coloraxis_showscale=False, margin=dict(t=20)
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.info("**csat_score** est la variable la plus déterminante (0,159). "
            "**login_per_month** (feature engineerée) se place en 3e position, "
            "devant **monthly_logins** brut — ce qui valide le feature engineering.")

    st.subheader("Analyse SHAP — XGBoost")
    st.markdown("""
    SHAP explique **pourquoi** le modèle produit chaque prédiction.
    - Valeur SHAP positive → pousse vers le churn
    - Valeur SHAP négative → réduit le risque de churn
    """)

    with st.spinner("Calcul des valeurs SHAP..."):
        xgb_model = models['XGBoost']
        explainer  = shap.TreeExplainer(xgb_model)
        sample     = X_test.sample(300, random_state=42)
        shap_vals  = explainer.shap_values(sample)

    fig_bar, ax1 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, sample, plot_type='bar',
                      max_display=15, show=False)
    plt.title("SHAP — Importance globale (XGBoost)", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_bar)

    fig_bee, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, sample, plot_type='dot',
                      max_display=12, show=False)
    plt.title("SHAP — Direction des effets", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_bee)

    st.info("**Lecture :** Un csat_score élevé (rouge) à gauche réduit le risque de churn. "
            "Des payment_failures élevés (rouge) à droite augmentent le risque.")



# PRÉDICTION CLIENT

elif page == "🎯 Prédiction client":
    st.title("Prédiction de churn — Client individuel")
    st.markdown("Renseignez les caractéristiques d'un client pour obtenir "
                "sa probabilité de churn en temps réel.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profil client")
        tenure_months  = st.slider("Ancienneté (mois)", 1, 72, 12)
        monthly_logins = st.slider("Connexions mensuelles", 0, 40, 10)
        csat_score     = st.slider("Score satisfaction (CSAT)", 1.0, 5.0, 3.0, 0.5)
        nps_score      = st.slider("NPS Score", -100, 100, 0)
        gender         = st.selectbox("Genre", ["Male", "Female"])
        customer_segment = st.selectbox("Segment client",
                                        ["Individual", "SME", "Enterprise"])

    with col2:
        st.subheader("Données financières")
        monthly_fee      = st.selectbox("Frais mensuels (€)",
                                        [10, 20, 30, 50, 75, 100, 150], index=2)
        total_revenue    = st.number_input("Revenu total (€)", 10, 5000, 360)
        payment_failures = st.slider("Échecs de paiement", 0, 5, 0)
        contract_type    = st.selectbox("Type de contrat",
                                        ["Monthly", "Yearly", "Two-Year"])
        payment_method   = st.selectbox("Moyen de paiement",
                                        ["Card", "PayPal", "BankTransfer"])
        discount_applied = st.selectbox("Remise appliquée", ["No", "Yes"])
        price_increase_last_3m = st.selectbox("Hausse tarifaire récente (3 mois)",
                                              ["No", "Yes"])

    with col3:
        st.subheader("Comportement")
        weekly_active_days = st.slider("Jours actifs / semaine", 0, 7, 3)
        avg_session_time   = st.slider("Durée session moy. (min)", 1.0, 30.0, 15.0)
        support_tickets    = st.slider("Tickets support", 0, 5, 1)
        last_login_days    = st.slider("Dernière connexion (jours)", 0, 30, 5)
        signup_channel     = st.selectbox("Canal d'inscription",
                                          ["Web", "Mobile", "Partner"])
        complaint_type     = st.selectbox("Type de plainte",
                                          ["No_complaint", "Billing",
                                           "Technical", "Service"])
        survey_response    = st.selectbox("Réponse enquête",
                                          ["Neutral", "Satisfied", "Unsatisfied"])

    if st.button("🔮 Calculer la probabilité de churn", type="primary"):

        # Construction du vecteur features 
        row_scaled = build_feature_vector(
            tenure_months, monthly_logins, csat_score, nps_score,
            monthly_fee, total_revenue, payment_failures,
            weekly_active_days, avg_session_time, support_tickets,
            last_login_days, gender, customer_segment, signup_channel,
            contract_type, payment_method, discount_applied,
            price_increase_last_3m, complaint_type, survey_response
        )

        # Prédiction 
        proba = active_model.predict_proba(row_scaled)[0][1]
        pred  = int(proba >= 0.5)

        st.markdown("---")

        # Résultat textuel 
        if pred == 1:
            st.error(f" Client à HAUT RISQUE de churn — "
                     f"Probabilité : **{proba*100:.1f}%**")
        elif proba >= 0.3:
            st.warning(f" Risque MODÉRÉ de churn — "
                       f"Probabilité : **{proba*100:.1f}%**")
        else:
            st.success(f" Risque FAIBLE de churn — "
                       f"Probabilité : **{proba*100:.1f}%**")

        # Jauge 
        gauge_color = ("#E53935" if proba > 0.5
                       else "#FF9800" if proba > 0.3
                       else "#4CAF50")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(proba * 100, 1),
            delta={'reference': 10.2, 'suffix': '% (moy.)'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': gauge_color},
                'steps': [
                    {'range': [0,  30],  'color': '#E8F5E9'},
                    {'range': [30, 50],  'color': '#FFF3E0'},
                    {'range': [50, 100], 'color': '#FFEBEE'},
                ],
                'threshold': {
                    'line':      {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value':     50
                }
            },
            title={'text': "Probabilité de churn (%)"}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        #  Détail des features engineerées 
        with st.expander("Détail des features calculées"):
            st.write({
                'login_per_month':
                    round(monthly_logins / (tenure_months + 1), 3),
                'payment_risk':
                    round(payment_failures * monthly_fee, 2),
                'recency_risk':
                    round(last_login_days / (avg_session_time + 1), 3),
            })

        # Recommandations 
        st.subheader("Recommandations")
        recs = []
        if payment_failures > 0:
            recs.append("Échecs de paiement détectés — "
                        "Contacter le client pour régulariser la situation.")
        if csat_score < 3.0:
            recs.append("Satisfaction client faible (CSAT < 3) — "
                        "Proposer un appel de suivi ou un geste commercial.")
        if tenure_months < 6:
            recs.append("Client récent (< 6 mois) — "
                        "Renforcer l'onboarding et l'accompagnement.")
        if monthly_logins < 5:
            recs.append("Engagement faible — "
                        "Lancer une campagne de réengagement personnalisée.")
        if last_login_days > 14:
            recs.append("Inactivité récente — "
                        "Relancer avec une offre exclusive.")
        if survey_response == "Unsatisfied":
            recs.append("Enquête négative — "
                        "Escalader au service client en priorité.")
        if contract_type == "Monthly" and proba > 0.3:
            recs.append("Contrat mensuel sans engagement — "
                        "Proposer une offre annuelle avec remise.")
        if not recs:
            recs.append("Profil stable — "
                        "Maintenir la relation et surveiller l'évolution du CSAT.")
        for r in recs:
            st.markdown(r)

        # Revenu à risque 
        st.markdown("---")
        rev_risk = round(monthly_fee * proba, 2)
        st.metric("Revenu mensuel à risque pour ce client",
                  f"{rev_risk} €",
                  f"{proba*100:.1f}% × {monthly_fee} €/mois")
