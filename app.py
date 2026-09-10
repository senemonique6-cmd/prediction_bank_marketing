
import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
import os 
# ==============================
# CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Prédiction bancaire",
    page_icon="🏦",
    layout="wide"
)

# ==============================
# CHARGEMENT DES MODÈLES
# ==============================

@st.cache_resource
def load_models():
    encoders = jb.load("encoders_bank.joblib")
    scaler = jb.load("scaler_bank.joblib")
    gb = jb.load("gb_model_bank.joblib")
    uniques = jb.load("uniques_bank.joblib")
    return encoders, scaler, gb, uniques


encoders, scaler, gb, uniques = load_models()


# ==============================
# FONCTION DE PRÉDICTION
# ==============================

def Pred_func(
    age, job, marital, education, housing, loan,
    contact, month, day_of_week, duration,
    campaign, pdays, previous, poutcome
):

    job = encoders[0].transform([job])[0]
    marital = encoders[1].transform([marital])[0]
    education = encoders[2].transform([education])[0]
    housing = encoders[3].transform([housing])[0]
    loan = encoders[4].transform([loan])[0]
    contact = encoders[5].transform([contact])[0]
    month = encoders[6].transform([month])[0]
    day_of_week = encoders[7].transform([day_of_week])[0]
    poutcome = encoders[8].transform([poutcome])[0]

    x_new = np.array([
        age,
        job,
        marital,
        education,
        housing,
        loan,
        contact,
        month,
        day_of_week,
        duration,
        campaign,
        pdays,
        previous,
        poutcome
    ], dtype=float)

    x_new = x_new.reshape(1, -1)

    x_new = scaler.transform(x_new)

    y_pred = gb.predict(x_new)

    return "yes" if y_pred[0] == 1 else "no"


# ==============================
# FONCTION DE PRÉDICTION CSV
# ==============================

def Pred_func_csv(df):

    predictions = []

    for _, row in df.iterrows():

        prediction = Pred_func(
            row["age"],
            row["job"],
            row["marital"],
            row["education"],
            row["housing"],
            row["loan"],
            row["contact"],
            row["month"],
            row["day_of_week"],
            row["duration"],
            row["campaign"],
            row["pdays"],
            row["previous"],
            row["poutcome"]
        )

        predictions.append(prediction)

    result_df = df.copy()
    result_df["y"] = predictions

    return result_df


# ==============================
# TITRE
# ==============================

st.title("🏦 Prédiction de souscription à un dépôt à terme")

st.write(
    """
    Cette application permet de prédire si un client est susceptible
    de souscrire ou non à un dépôt à terme.
    """
)


# ==============================
# ONGLETS
# ==============================

tab1, tab2 = st.tabs([
    "🔮 Prédiction simple",
    "📄 Prédiction multiple CSV"
])


# ==============================
# PRÉDICTION SIMPLE
# ==============================

with tab1:

    st.subheader("Informations du client")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Âge",
            min_value=18,
            max_value=100,
            value=30
        )

        job = st.selectbox(
            "Profession",
            encoders[0].classes_.tolist()
        )

        marital = st.selectbox(
            "État matrimonial",
            encoders[1].classes_.tolist()
        )

        education = st.selectbox(
            "Niveau d'éducation",
            encoders[2].classes_.tolist()
        )

        housing = st.selectbox(
            "Prêt immobilier",
            encoders[3].classes_.tolist()
        )

    with col2:

        loan = st.selectbox(
            "Prêt personnel",
            encoders[4].classes_.tolist()
        )

        contact = st.selectbox(
            "Type de contact",
            encoders[5].classes_.tolist()
        )

        month = st.selectbox(
            "Mois",
            encoders[6].classes_.tolist()
        )

        day_of_week = st.selectbox(
            "Jour de la semaine",
            encoders[7].classes_.tolist()
        )

        duration = st.number_input(
            "Durée du dernier appel",
            min_value=0,
            value=100
        )

    with col3:

        campaign = st.number_input(
            "Nombre de contacts pendant la campagne",
            min_value=0,
            value=1
        )

        pdays = st.number_input(
            "Nombre de jours depuis le dernier contact",
            min_value=0,
            value=999
        )

        previous = st.number_input(
            "Nombre de contacts précédents",
            min_value=0,
            value=0
        )

        poutcome = st.selectbox(
            "Résultat de la campagne précédente",
            encoders[8].classes_.tolist()
        )

    st.divider()

    if st.button(
        "🔍 Faire la prédiction",
        type="primary",
        use_container_width=True
    ):

        prediction = Pred_func(
            age,
            job,
            marital,
            education,
            housing,
            loan,
            contact,
            month,
            day_of_week,
            duration,
            campaign,
            pdays,
            previous,
            poutcome
        )

        st.subheader("Résultat")

        if prediction == "yes":
            st.success(
                "✅ Le modèle prédit que le client est susceptible "
                "de souscrire au dépôt à terme."
            )
        else:
            st.warning(
                "❌ Le modèle prédit que le client ne devrait "
                "pas souscrire au dépôt à terme."
            )


# ==============================
# PRÉDICTION CSV
# ==============================

with tab2:

    st.subheader("Prédiction multiple à partir d'un fichier CSV")

    st.write(
        """
        Importez un fichier CSV contenant les variables nécessaires
        à la prédiction.
        """
    )

    uploaded_file = st.file_uploader(
        "Importer votre fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.write("### Aperçu du fichier")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            required_columns = [
                "age",
                "job",
                "marital",
                "education",
                "housing",
                "loan",
                "contact",
                "month",
                "day_of_week",
                "duration",
                "campaign",
                "pdays",
                "previous",
                "poutcome"
            ]

            missing_columns = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:

                st.error(
                    "❌ Colonnes manquantes : "
                    + ", ".join(missing_columns)
                )

            else:

                if st.button(
                    "🔍 Lancer les prédictions",
                    type="primary"
                ):

                    with st.spinner("Prédiction en cours..."):

                        result_df = Pred_func_csv(df)

                    st.success(
                        "✅ Les prédictions ont été effectuées."
                    )

                    st.write("### Résultats")

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    csv = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Télécharger les prédictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                f"Une erreur est survenue : {e}"
            )
```
