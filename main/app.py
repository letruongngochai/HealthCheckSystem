from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from CONSTANT import *
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from src.data.data_loading import BasicDataset
from src.models.unet import UNet
from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import os
from google.cloud import storage
import requests
import uuid


model_path = 'src/models/svm.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

sc_path = 'src/models/sc.pkl'
with open(sc_path, 'rb') as file:
    sc = pickle.load(file)

seg_model = r"src/models/checkpoint_epoch50.pth"
net = UNet(n_channels=1, n_classes=1, bilinear=True)
net = net.to(memory_format=torch.channels_last)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
state_dict = torch.load(seg_model, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)

def upload_image_to_gcs(image: Image.Image, destination_blob_name: str) -> str:
    """Upload ảnh PIL lên GCS và trả về URL công khai"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    # Chuyển ảnh PIL thành bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    blob.upload_from_file(img_bytes, content_type="image/png")
    blob.make_public()

    return blob.public_url

BUCKET_NAME = 'model_result_ngochai'

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


# Create FastAPI instance
app = FastAPI()
data = pd.read_csv("src/data/heart.csv")

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

# class ImgResponse(BaseModel):
#     image: Image

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


class SegmentRequest(BaseModel):
    image_url: str

@app.post("/segment")
async def segment_image(image: SegmentRequest):
    # Step 1: Download ảnh từ image_url
    response = requests.get(image.image_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Could not download image from provided URL")

    img = Image.open(io.BytesIO(response.content)).convert("L")
    resized_img = img.resize((256, 256))

    # Step 2: Dự đoán segmentation mask
    mask = predict_img(
        net=net,
        full_img=resized_img,
        scale_factor=0.5,
        out_threshold=0.5,
        device=device
    )

    # Step 3: Chuyển mask thành ảnh kết quả
    segmented_img = mask_to_image(mask, mask_values)

    # Step 4: Upload ảnh kết quả lên GCS
    filename = f"results/{uuid.uuid4()}_segmented.png"
    segmented_url = upload_image_to_gcs(segmented_img, filename)

    return {
        "status": "success",
        "segmented_url": segmented_url
    }
