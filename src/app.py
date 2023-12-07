import json
import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

class StockData(BaseModel):
    stock_id: int
    date_id: int
    seconds_in_bucket: int
    imbalance_size: float
    imbalance_buy_sell_flag: int
    reference_price: float
    matched_size: float
    far_price: Optional[float] = None
    near_price: Optional[float] = None
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    wap: float
    time_id: int
    row_id: str

app = FastAPI()

scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('.')

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/predict")
def predict(data: StockData):
    if 'far_price' in data:
        del data['far_price']

    if 'near_price' in data:
        del data['near_price']

    # gestion d'erreur necessaire ici
    keys_to_check = ["imbalance_size", "reference_price", "matched_size", "bid_price", "ask_price", "wap"]

    # Check if any key has an empty value
    if any(data[key] in [None, ''] for key in keys_to_check if key in data):
        return ValueError("Missing values for keys: {}".format(", ".join(key for key in keys_to_check if key in data)))

    data['imbalance_ratio'] = data['imbalance_size']/data['matched_size']

    data['spread_size'] = data['ask_size']-data['bid_size']
    data['spread_size_ratio'] = data['ask_size']/data['bid_size']

    data['bid_value'] = data['bid_price']*data['bid_size']
    data['ask_value'] = data['ask_price']*data['ask_size']

    data['spread_price'] = (data['ask_price']-data['bid_price'])/(data['ask_price'])

    data['wap_spread'] = ((data['bid_price']*data['bid_size'])+(data['ask_price']*data['ask_size']))/(data['bid_size']+data['ask_size'])

    df = pd.DataFrame([data])
    df = df.set_index('stock_id')

    scaled = scaler.transform(df)
    
    prediction = model.predict(scaled)

    return JSONResponse(prediction[0][0])