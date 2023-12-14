import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional
import json
from sklearn.preprocessing import StandardScaler


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

def preprocess(df):
    data = pd.DataFrame([df.dict()])
    # data cleaning
    data=data.drop(['far_price','near_price', 'row_id', 'time_id', 'date_id'],axis=1)

#feature engineering
    data['imbalance_ratio'] = data['imbalance_size']/data['matched_size']
    data['spread_size'] = data['ask_size']-data['bid_size']
    data['spread_size_ratio'] = data['ask_size']/data['bid_size']
    data['bid_value'] = data['bid_price']*data['bid_size']
    data['ask_value'] = data['ask_price']*data['ask_size']
    data['spread_price'] = (data['ask_price']-data['bid_price'])/(data['ask_price'])
    data['wap_spread'] = ((data['bid_price']*data['bid_size'])+(data['ask_price']*data['ask_size']))/(data['bid_size']+data['ask_size'])

    data=data.drop(['matched_size','stock_id'],axis=1)

    scaled = scaler.transform(data)

    return scaled

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(data: StockData):
# def predict(data):

    df =  preprocess(data)

    # gestion d'erreur necessaire ici
    # keys_to_check = ["imbalance_size", "reference_price", "matched_size", "bid_price", "ask_price", "wap"]

    # # Check if any key has an empty value
    # if any(data[key] in [None, ''] for key in keys_to_check if key in data):
    #     return ValueError("Missing values for keys: {}".format(", ".join(key for key in keys_to_check if key in data)))

    prediction = model.predict(df)
    prediction_list = prediction.tolist()


    return JSONResponse(content=json.dumps(dict(enumerate(prediction_list))))