from flask import Flask, request, jsonify
import pickle
import catboost as cb
import pandas as pd
import numpy as np
from sklearn import LinearRegressor, train_test_split

app = Flask(__name__)

model_path = "testmodel.pkl"  # Assurez-vous d'avoir le bon chemin vers votre modèle CatBoost
with open(model_path, "rb") as model_file:
    catboost_model = pickle.load(model_file)

def preprocess(df):
    df_pd = pd.read_csv(df)
    # Ajoutez ici des étapes de prétraitement si nécessaire
    return df_pd

@app.route('/')
def welcome():
    return 'Bienvenue'

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Obtenez les données d'entrée de la requête GET
        data = request.args.get('data')

        # Prétraitement des données
        data = preprocess(data)

        # Faire la prédiction avec le modèle CatBoost
        prediction = catboost_model.predict(data)

        # Retournez la prédiction au format JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # En cas d'erreur, retournez un message d'erreur au format JSON
        return jsonify({'error': str(e)})

# Point d'entrée pour l'application Flask
if __name__ == '__main__':
    # Lancer l'application sur le port 5001
    app.run(host='0.0.0.0', port=5001)

#http://127.0.0.1:5001 IP