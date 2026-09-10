import gradio as gr
import joblib as jb
import pandas as pd
import numpy as np

# ==============================
# CHARGEMENT DES OBJETS ENTRAÎNÉS
# ==============================
# Ces 4 fichiers .joblib doivent être dans le même dossier que app.py :
#   - encoders_bank.joblib
#   - scaler_bank.joblib
#   - gb_model_bank.joblib
#   - uniques_bank.joblib

encoders = jb.load('encoders_bank.joblib')
scaler = jb.load('scaler_bank.joblib')
gb = jb.load('gb_model_bank.joblib')
uniques = jb.load('uniques_bank.joblib')


# ==============================
# FONCTION DE PRÉDICTION SIMPLE
# ==============================
def Pred_func(age, job, marital, education, housing, loan,
              contact, month, day_of_week, duration,
              campaign, pdays, previous, poutcome):

    # Encoder les variables catégorielles
    job = encoders[0].transform([job])[0]
    marital = encoders[1].transform([marital])[0]
    education = encoders[2].transform([education])[0]
    housing = encoders[3].transform([housing])[0]
    loan = encoders[4].transform([loan])[0]
    contact = encoders[5].transform([contact])[0]
    month = encoders[6].transform([month])[0]
    day_of_week = encoders[7].transform([day_of_week])[0]
    poutcome = encoders[8].transform([poutcome])[0]

    # Créer le vecteur des variables
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
    ])

    # Convertir en tableau 2D
    x_new = x_new.reshape(1, -1)

    # Normaliser les données
    x_new = scaler.transform(x_new)

    # Prédiction
    y_pred = gb.predict(x_new)

    # Retourner la classe
    if y_pred[0] == 1:
        return "yes"
    else:
        return "no"


# ==============================
# FONCTION DE PRÉDICTION MULTIPLE (CSV)
# ==============================
def Pred_func_csv(file):

    df = pd.read_csv(file)

    predictions = []

    for _, row in df.iterrows():
        y_pred = Pred_func(
            row['age'],
            row['job'],
            row['marital'],
            row['education'],
            row['housing'],
            row['loan'],
            row['contact'],
            row['month'],
            row['day_of_week'],
            row['duration'],
            row['campaign'],
            row['pdays'],
            row['previous'],
            row['poutcome']
        )
        predictions.append(y_pred)

    df['y'] = predictions
    df.to_csv('predictions.csv', index=False)

    return 'predictions.csv'


# ==============================
# DÉFINITION DE L'INTERFACE
# ==============================
demo = gr.Blocks(theme='shivi/calm_seafoam')

# --- Interface 1 : Prédiction simple ---
inputs = [
    gr.Number(label='Âge'),
    gr.Dropdown(choices=encoders[0].classes_.tolist(), label='Profession'),
    gr.Dropdown(choices=encoders[1].classes_.tolist(), label='État matrimonial'),
    gr.Dropdown(choices=encoders[2].classes_.tolist(), label="Niveau d'éducation"),
    gr.Dropdown(choices=encoders[3].classes_.tolist(), label='Prêt immobilier'),
    gr.Dropdown(choices=encoders[4].classes_.tolist(), label='Prêt personnel'),
    gr.Dropdown(choices=encoders[5].classes_.tolist(), label='Type de contact'),
    gr.Dropdown(choices=encoders[6].classes_.tolist(), label='Mois'),
    gr.Dropdown(choices=encoders[7].classes_.tolist(), label='Jour de la semaine'),
    gr.Number(label='Durée du dernier appel'),
    gr.Number(label='Nombre de contacts pendant la campagne'),
    gr.Number(label='Nombre de jours depuis le dernier contact'),
    gr.Number(label='Nombre de contacts précédents'),
    gr.Dropdown(choices=encoders[8].classes_.tolist(), label='Résultat de la campagne précédente'),
]

outputs = gr.Textbox(label="Prédiction")

interface1 = gr.Interface(
    fn=Pred_func,
    inputs=inputs,
    outputs=outputs,
    title="Prédire la souscription à un dépôt à terme",
    description="""
    Ce modèle permet de prédire si un client souscrira ou non
    à un dépôt à terme à partir de ses caractéristiques
    et des informations relatives aux campagnes de marketing.
    """
)

# --- Interface 2 : Prédiction multiple via CSV ---
interface2 = gr.Interface(
    fn=Pred_func_csv,
    inputs=gr.File(label='Importer un fichier CSV', file_types=['.csv']),
    outputs=gr.File(label='Télécharger le fichier avec les prédictions'),
    title="Prédiction multiple à partir d'un fichier CSV",
    description="""
    Importez un fichier CSV contenant les variables nécessaires
    à la prédiction. Le modèle ajoutera une colonne contenant
    la prédiction de souscription au dépôt à terme.
    """
)

with demo:
    gr.TabbedInterface(
        [interface1, interface2],
        ['Prédiction simple', 'Prédiction multiple']
    )


# ==============================
# LANCEMENT DE L'INTERFACE
# ==============================
if __name__ == "__main__":
    demo.launch()
