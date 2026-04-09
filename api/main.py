from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

from preprocessing.pipeline import preprocess

app = FastAPI()

# Load model and vectorizer
MODEL_PATH = os.path.join("model", "xgb_model.joblib")
model = joblib.load(MODEL_PATH)

class PostText(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Welcome to the Tigrinya Harmful Post Detector API"}

@app.post("/predict")
def predict(post: PostText):
    features = preprocess(post.text)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]

    return {
        "text": post.text,
        "prediction": "Harmful" if pred == 1 else "Neutral",
        "confidence": round(float(prob), 4)
    }
