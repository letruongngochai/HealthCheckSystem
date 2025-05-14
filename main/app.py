from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from CONSTANT import *
import pickle
import numpy as np

model_path = 'src\models\svm.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

sc_path = 'src\models\sc.pkl'
with open(sc_path, 'rb') as file:
    sc = pickle.load(file)

# Create FastAPI instance
app = FastAPI()
data = pd.read_csv("src\data\heart.csv")

class Patient(BaseModel):
    age: int
    sex: str
    cp: str
    trestbps: int
    chol: int
    fbs: str
    restecg: str
    thalach: int
    exang: str
    oldpeak: float
    slope: str
    ca: int
    thal: str

    #sample
    # "age": 63,
    # "sex": "Female",
    # "cp": "Asymptomatic",
    # "trestbps": 145,
    # "chol": 233,
    # "fbs": "True (> 120 mg/dl)",
    # "restecg": "Normal",
    # "thalach": 150,
    # "exang": "No",
    # "oldpeak": 2.3,
    # "slope": "Upsloping",
    # "ca": 0,
    # "thal": "Fixed defect"

class PredictionResponse(BaseModel):
    prediction: int

# GET method - Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict/", response_model=PredictionResponse)
def predict(patient: Patient):
    model_age = patient.age
    model_sex = sex_mapping[patient.sex]
    model_cp = chest_pain_type_mapping[patient.cp]
    model_trestbps = patient.trestbps
    model_chol = patient.chol
    model_fbs = fasting_blood_sugar_mapping[patient.fbs]
    model_restecg = resting_ecg_mapping[patient.restecg]
    model_thalach = patient.thalach
    model_exang = exercise_induced_angina_mapping[patient.exang]
    model_oldpeak = patient.oldpeak
    model_slope = st_slope_mapping[patient.slope]
    model_ca = patient.ca
    model_thal = thal_mapping[patient.thal]

    num_features = 10 + 3 + 2 + 3  # Numerical + CP + restecg + thal
    input_features = np.zeros((1, num_features))

    input_features[0, 0] = model_age
    input_features[0, 1] = model_sex
    input_features[0, 2] = model_trestbps
    input_features[0, 3] = model_chol
    input_features[0, 4] = model_fbs
    input_features[0, 5] = model_thalach
    input_features[0, 6] = model_exang
    input_features[0, 7] = model_oldpeak
    input_features[0, 8] = model_slope
    input_features[0, 9] = model_ca


    if model_cp == 0:
        input_features[0, 10:13] = [0, 0, 0]
    elif model_cp == 1:
        input_features[0, 10:13] = [1, 0, 0]
    elif model_cp == 2:
        input_features[0, 10:13] = [0, 1, 0]
    elif model_cp == 3:
        input_features[0, 10:13] = [0, 0, 1]


    if model_restecg == 0:
        input_features[0, 13:15] = [0, 0]
    elif model_restecg == 1:
        input_features[0, 13:15] = [1, 0]
    elif model_restecg == 2:
        input_features[0, 13:15] = [0, 1]


    if model_thal == 0:
        input_features[0, 15:18] = [0, 0, 0]
    elif model_thal == 1:
        input_features[0, 15:18] = [1, 0, 0]
    elif model_thal == 2:
        input_features[0, 15:18] = [0, 1, 0]
    elif model_thal == 3:
        input_features[0, 15:18] = [0, 0, 1]
    input_features = sc.transform(input_features)
    print(input_features)
    pred = model.predict(input_features) 
    return {"prediction": int(pred[0])}

@app.get("/data/")
def get_data():
    df = pd.read_csv("src/data/heart.csv")
    return df.to_dict(orient="records")

# # GET method - Get item by ID
# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

# POST method - Create an item
# @app.post("/items/")
# def create_item(item: Item):
#     return {"item_name": item.name, "item_price": item.price, "is_offer": item.is_offer}

# # POST method - Simple echo endpoint
# @app.post("/echo/")
# def echo_message(message: dict):
#     return {"echoed_message": message}

# To run the application:
# uvicorn app:app --reload
# Replace 'main' with your filename if different