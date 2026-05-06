import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              classification_report)
import warnings, os
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION PAGE
# ============================================================
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CHARGEMENT DES MODÈLES ET DONNÉES (mis en cache)
# ============================================================
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = {
        'Régression Logistique': joblib.load(f'{base}/models/logistic_regression.pkl'),
        'Random Forest':         joblib.load(f'{base}/models/random_forest.pkl'),
        'XGBoost':               joblib.load(f'{base}/models/xgboost.pkl'),
    }
    scaler = joblib.load(f'{base}/data/processed/scaler.pkl')
    return models, scaler

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X_test  = pd.read_csv(f'{base}/data/processed/X_test.csv')
    y_test  = pd.read_csv(f'{base}/data/processed/y_test.csv').squeeze()
    X_train = pd.read_csv(f'{base}/data/processed/X_train.csv')
    y_train = pd.read_csv(f'{base}/data/processed/y_train.csv').squeeze()
    df_raw  = pd.read_csv(f'{base}/data/raw/customer_churn.csv')
    return X_test, y_test, X_train, y_train, df_raw

models, scaler = load_models()
X_test, y_test, X_train, y_train, df_raw = load_data()

# ============================================================
# SIDEBAR — Navigation
# ============================================================
st.sidebar.title("📊 Churn Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Vue d'ensemble", "📈 Performance modèles",
     "🔍 Interprétabilité", "🎯 Prédiction client"]
)
st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "Modèle actif",
    list(models.keys()),
    index=1  # Random Forest par défaut
)
active_model = models[model_choice]

# ============================================================
# PAGE 1 — VUE D'ENSEMBLE
# ============================================================
if page == "🏠 Vue d'ensemble":
    st.title("🏠 Vue d'ensemble — Rétention Client")
    st.markdown("Plateforme décisionnelle de prédiction du churn et d'analyse du risque de revenus.")

    # KPI Cards
    y_pred_all  = active_model.predict(X_test)
    y_proba_all = active_model.predict_proba(X_test)[:, 1]
    n_churn     = int(y_pred_all.sum())
    df_raw_copy = df_raw.copy()
    mean_monthly_fee = df_raw_copy['monthly_fee'].mean()
    revenue_at_risk  = round(n_churn * mean_monthly_fee, 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients analysés",    f"{len(X_test):,}")
    col2.metric("Clients à risque",    f"{n_churn:,}",
                f"{n_churn/len(X_test)*100:.1f}% du portefeuille")
    col3.metric("Revenu mensuel à risque",
                f"{revenue_at_risk:,.0f} €",
                "Estimation conservatrice")
    col4.metric("Modèle actif", model_choice.split()[0] + "...",
                "Changer dans la barre latérale")

    st.markdown("---")

    # Distribution du churn dans les données brutes
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
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20,b=20))
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
                           coloraxis_showscale=False, margin=dict(t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution des probabilités prédites
    st.subheader("Distribution des probabilités de churn prédites")
    fig3 = px.histogram(
        x=y_proba_all,
        nbins=50,
        color_discrete_sequence=['#7B1FA2'],
        labels={'x': 'Probabilité de churn', 'y': 'Nombre de clients'}
    )
    fig3.add_vline(x=0.5, line_dash='dash', line_color='red',
                   annotation_text='Seuil 0.5')
    fig3.update_layout(height=300, margin=dict(t=20,b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # Top clients à risque
    st.subheader("Top 10 clients à risque les plus élevé")
    risk_df = X_test.copy()
    risk_df['proba_churn'] = y_proba_all
    risk_df['prediction']  = y_pred_all
    risk_df['vrai_label']  = y_test.values
    top_risk = risk_df.nlargest(10, 'proba_churn')[
        ['proba_churn', 'csat_score', 'tenure_months',
         'monthly_logins', 'payment_failures']
    ].round(3)
    top_risk['proba_churn'] = top_risk['proba_churn'].apply(
        lambda x: f"{x*100:.1f}%"
    )
    st.dataframe(top_risk, use_container_width=True)

# ============================================================
# PAGE 2 — PERFORMANCE MODÈLES
# ============================================================
elif page == "📈 Performance modèles":
    st.title("📈 Comparaison des performances")

    # Tableau comparatif
    perf_data = {
        'Modèle':     ['Régression Logistique', 'Random Forest', 'XGBoost', 'MLP'],
        'Accuracy':   [0.6895, 0.8310, 0.8160, 0.7575],
        'Precision':  [0.1983, 0.2781, 0.2657, 0.2384],
        'Recall':     [0.6716, 0.4118, 0.4559, 0.6275],
        'F1-Score':   [0.3061, 0.3320, 0.3357, 0.3455],
        'ROC-AUC':    [0.7510, 0.7908, 0.7728, 0.7554],
    }
    df_perf = pd.DataFrame(perf_data).set_index('Modèle')

    st.subheader("Tableau comparatif — toutes métriques")
    st.dataframe(
        df_perf.style.highlight_max(axis=0, color='#C8E6C9')
                     .highlight_min(axis=0, color='#FFCDD2')
                     .format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("""
    > Vert = meilleur score | Rouge = score le plus faible pour cette métrique.
    > Le Random Forest domine sur ROC-AUC. Le MLP a le meilleur F1. 
    > La Régression Logistique a le meilleur Recall — utile si on veut minimiser les faux négatifs.
    """)

    # Courbes ROC
    st.subheader("Courbes ROC — comparaison")
    fig_roc = go.Figure()
    colors_roc = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for (name, model), color in zip(models.items(), colors_roc):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score   = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"{name} (AUC={auc_score:.3f})",
            line=dict(color=color, width=2)
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        name='Baseline (AUC=0.5)',
        line=dict(color='gray', dash='dash')
    ))
    fig_roc.update_layout(
        xaxis_title='Taux de faux positifs',
        yaxis_title='Taux de vrais positifs (Recall)',
        height=450, margin=dict(t=20)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Matrice de confusion du modèle actif
    st.subheader(f"Matrice de confusion — {model_choice}")
    y_pred = active_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Churn', 'Churn'],
                yticklabels=['Non-Churn', 'Churn'], ax=ax)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title(f'Matrice de confusion — {model_choice}')
    st.pyplot(fig_cm)

    tn, fp, fn, tp = cm.ravel()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vrais Positifs (churners détectés)", tp)
    col2.metric("Faux Négatifs (churners ratés)", fn,
                delta=f"-{fn} clients perdus", delta_color="inverse")
    col3.metric("Vrais Négatifs", tn)
    col4.metric("Faux Positifs", fp)

# ============================================================
# PAGE 3 — INTERPRÉTABILITÉ
# ============================================================
elif page == "🔍 Interprétabilité":
    st.title("🔍 Interprétabilité des modèles")

    # Feature Importance Random Forest
    st.subheader("Feature Importance — Random Forest")
    rf_model = models['Random Forest']
    feat_imp = pd.Series(
        rf_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(15)

    fig_fi = px.bar(
        x=feat_imp.values,
        y=feat_imp.index,
        orientation='h',
        color=feat_imp.values,
        color_continuous_scale=['#B5D4F4', '#0C447C'],
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    fig_fi.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500, coloraxis_showscale=False,
        margin=dict(t=20)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info("""
    **Lecture :** `csat_score` est la variable la plus déterminante (score de satisfaction client).
    `tenure_months` (ancienneté) et `login_per_month` (engagement normalisé) suivent.
    `payment_failures` confirme que les échecs de paiement sont un signal critique de résiliation.
    """)

    # SHAP
    st.subheader("Analyse SHAP — XGBoost")
    st.markdown("""
    SHAP permet d'expliquer **pourquoi** le modèle produit chaque prédiction.
    - Une valeur SHAP positive pousse vers le churn
    - Une valeur SHAP négative réduit le risque de churn
    """)

    with st.spinner("Calcul des valeurs SHAP en cours..."):
        xgb_model  = models['XGBoost']
        explainer  = shap.TreeExplainer(xgb_model)
        sample     = X_test.sample(300, random_state=42)
        shap_vals  = explainer.shap_values(sample)

    fig_shap, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, sample, plot_type='bar',
                      max_display=15, show=False)
    plt.title("SHAP — Importance globale (XGBoost)", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_shap)

    fig_bee, ax2 = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, sample, plot_type='dot',
                      max_display=12, show=False)
    plt.title("SHAP — Direction des effets", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_bee)

    st.info("""
    **Lecture du beeswarm :**
    - Points rouges (valeur élevée) à gauche = cette feature élevée réduit le churn
    - Points rouges à droite = cette feature élevée augmente le churn
    Par exemple : `csat_score` élevé (rouge) à gauche → satisfaction élevée protège contre le churn.
    """)

# ============================================================
# PAGE 4 — PRÉDICTION CLIENT EN TEMPS RÉEL
# ============================================================
elif page == "🎯 Prédiction client":
    st.title("🎯 Prédiction de churn — Client individuel")
    st.markdown("Renseignez les caractéristiques d'un client pour obtenir sa probabilité de churn.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profil client")
        tenure_months   = st.slider("Ancienneté (mois)", 1, 72, 12)
        monthly_logins  = st.slider("Connexions mensuelles", 0, 40, 10)
        csat_score      = st.slider("Score satisfaction (CSAT)", 1.0, 5.0, 3.0, 0.5)
        nps_score       = st.slider("NPS Score", -100, 100, 0)

    with col2:
        st.subheader("Données financières")
        monthly_fee      = st.selectbox("Frais mensuels (€)", [10,20,30,50,75,100,150], index=2)
        total_revenue    = st.number_input("Revenu total (€)", 10, 5000, 360)
        payment_failures = st.slider("Échecs de paiement", 0, 5, 0)
        contract_type    = st.selectbox("Type de contrat", ['Monthly','Yearly','Two-Year'])

    with col3:
        st.subheader("Comportement")
        weekly_active_days = st.slider("Jours actifs / semaine", 0, 7, 3)
        avg_session_time   = st.slider("Durée session moy. (min)", 1.0, 30.0, 15.0)
        support_tickets    = st.slider("Tickets support", 0, 5, 1)
        last_login_days    = st.slider("Dernière connexion (jours)", 0, 30, 5)

    if st.button("🔮 Calculer la probabilité de churn", type="primary"):
        # Construction d'un vecteur de features aligné sur X_train
        template = pd.DataFrame(0, index=[0], columns=X_train.columns)

        # Remplissage des valeurs numériques
        num_vals = {
            'tenure_months': tenure_months,
            'monthly_logins': monthly_logins,
            'csat_score': csat_score,
            'nps_score': nps_score,
            'monthly_fee': monthly_fee,
            'total_revenue': total_revenue,
            'payment_failures': payment_failures,
            'weekly_active_days': weekly_active_days,
            'avg_session_time': avg_session_time,
            'support_tickets': support_tickets,
            'last_login_days_ago': last_login_days,
            'login_per_month': monthly_logins / (tenure_months + 1),
            'payment_risk': payment_failures * monthly_fee,
            'recency_risk': last_login_days / (avg_session_time + 1),
        }
        for col, val in num_vals.items():
            if col in template.columns:
                template[col] = val

        # Encodage contrat
        if 'contract_type_Monthly' in template.columns and contract_type == 'Monthly':
            template['contract_type_Monthly'] = 1
        if 'contract_type_Yearly' in template.columns and contract_type == 'Yearly':
            template['contract_type_Yearly'] = 1

        # Prédiction avec le modèle actif
        proba = active_model.predict_proba(template)[0][1]
        pred  = int(proba >= 0.5)

        # Affichage du résultat
        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ Client à HAUT RISQUE de churn — Probabilité : **{proba*100:.1f}%**")
        elif proba >= 0.3:
            st.warning(f"🟡 Risque MODÉRÉ de churn — Probabilité : **{proba*100:.1f}%**")
        else:
            st.success(f"✅ Risque FAIBLE de churn — Probabilité : **{proba*100:.1f}%**")

        # Jauge visuelle
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(proba * 100, 1),
            delta={'reference': 10.2, 'suffix': '% (moyenne)'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E53935" if proba > 0.5 else
                                 "#FF9800" if proba > 0.3 else "#4CAF50"},
                'steps': [
                    {'range': [0, 30],  'color': '#E8F5E9'},
                    {'range': [30, 50], 'color': '#FFF3E0'},
                    {'range': [50, 100],'color': '#FFEBEE'}
                ],
                'threshold': {'line': {'color': 'red', 'width': 4},
                              'thickness': 0.75, 'value': 50}
            },
            title={'text': "Probabilité de churn (%)"}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Recommandations
        st.subheader("💡 Recommandations")
        rec = []
        if payment_failures > 0:
            rec.append("🔴 Echecs de paiement détectés — Contacter le client pour régulariser la situation.")
        if csat_score < 3:
            rec.append("🔴 Satisfaction client faible — Proposer un appel de suivi ou un geste commercial.")
        if tenure_months < 6:
            rec.append("🟡 Client récent (< 6 mois) — Renforcer l'onboarding et l'accompagnement.")
        if monthly_logins < 5:
            rec.append("🟡 Engagement faible — Envoyer une campagne de réengagement personnalisée.")
        if last_login_days > 14:
            rec.append("🟡 Inactivité récente — Relancer avec une offre exclusive.")
        if not rec:
            rec.append("✅ Profil stable — Maintenir la relation et surveiller l'évolution du CSAT.")
        for r in rec:
            st.markdown(r)