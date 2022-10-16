import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference
import pickle
import os
from typing import List

# Instantiate the app.
app = FastAPI()

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    
def initialize():
    path = os.path.abspath(os.getcwd())
    global model, encoder, lb, categorical_features
    encoder = pickle.load(open(path + "/model/encoder_random_forest.pkl", 'rb'))
    model = pickle.load(open(path + "/model/random_forest_classifier.pkl", 'rb'))
    lb = pickle.load(open(path + "/model/lb.pkl", 'rb'))
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


initialize()


class CensusRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    occupation: str
    relationship: str
    race: str
    sex: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                'age': 27,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Women',
                'capital-gain': 4000,
                'capital-loss': 0,
                'hours-per-week': 36,
                'native-country': 'Greece'
            },
            "example2": {
                'age': 15,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Women',
                'capital-gain': 9000,
                'capital-loss': 0,
                'hours-per-week': 36,
                'native-country': 'United-States'
            }
        }


class CensusBatchRequest(BaseModel):
    people: List[CensusRequest]


# Endpoint for welcoming
@app.get("/")
async def intial_page():
    return {"initial_message": "Welcome to the Census Salary Prediction Service built by Çağlar Çankaya"}


# Endpoint for making single requests

@app.post("/predict/")
async def predict_single(params: CensusRequest):
    df = pd.DataFrame(params.dict(by_alias=True), index=[1])
    X_test, _, _, _ = process_data(df, categorical_features=categorical_features, label=None,
                                   encoder=encoder, lb=lb, training=False)
    y_pred = inference(model, X_test)
    salary = {'salary': lb.inverse_transform(y_pred)[0]}
    return salary


# Endpoint for making batch requests
@app.post("/predict_batch/")
async def predict_batch(params: CensusBatchRequest):
    values = [dict(person) for person in params.people]

    df = pd.DataFrame(values)
    df.rename(columns={'marital_status': 'marital-status', 'native_country': 'native-country'}, inplace=True)
    X_test, _, _, _ = process_data(df, categorical_features=categorical_features, label=None,
                                   encoder=encoder, lb=lb, training=False)

    predictions = []
    for row in X_test:
        predictions.append(inference(model, row.reshape(1, -1)))

    salaries = []
    for prediction in predictions:
        salaries.append(lb.inverse_transform(prediction)[0])
    response = {'salary': salaries}
    return response


# For debugging, no need to include in build
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
