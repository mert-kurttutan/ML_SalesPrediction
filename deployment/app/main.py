import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist
import xgboost as xgb


app = FastAPI(title="Predicting Wine Class with batching")

# Represents a batch of item-store to predict sales for
class Sales(BaseModel):
    batches: List[conlist(item_type=float, min_items=37, max_items=37)]


@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    global reg_xgb
    reg_xgb = xgb.Booster()
    reg_xgb.load_model("/app/model.txt")


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now go to the port of local machine binded to docker container"


@app.post("/predict")
def predict(sales: Sales):
    batches = sales.batches
    
    # Turn this into Dmatrix as XGBoost accepts this format when loaded from
    # serialized files
    dm_batches = xgb.DMatrix(data=np.array(batches))
    pred = reg_xgb.predict(dm_batches).tolist()
    return {"Prediction": pred}
