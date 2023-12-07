from flask import Flask, request, jsonify
import pickle
import catboost as cb
import pandas as pd
import numpy as np
from sklearn import LinearRegressor, train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model_path = "testmodel.pkl"  # Assurez-vous d'avoir le bon chemin vers votre modèle CatBoost
with open(model_path, "rb") as model_file:
    catboost_model = pickle.load(model_file)

def preprocess(df):
    data = pd.read_csv(df)
    # data cleaning
    data=data.drop(['far_price','near_price', 'row_id', 'time_id', 'date_id'],axis=1)
    data.dropna(subset=['target'],inplace=True)

#feature engineering
    data['imbalance_ratio'] = data['imbalance_size']/data['matched_size']
    data['spread_size'] = data['ask_size']-data['bid_size']
    data['spread_size_ratio'] = data['ask_size']/data['bid_size']
    data['bid_value'] = data['bid_price']*data['bid_size']
    data['ask_value'] = data['ask_price']*data['ask_size']
    data['spread_price'] = (data['ask_price']-data['bid_price'])/(data['ask_price'])
    data['wap_spread'] = ((data['bid_price']*data['bid_size'])+(data['ask_price']*data['ask_size']))/(data['bid_size']+data['ask_size'])

    data=data.drop(['matched_size','stock_id'],axis=1)
    data = data.dropna()


    scaler = StandardScaler()
    X = scaler.fit_transform(X)



    return data




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