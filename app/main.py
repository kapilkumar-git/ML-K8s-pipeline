from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Iris Classifier API", version="1.0.0")

model = joblib.load("model.pkl")
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    pred_idx = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0].tolist()
    return {
        "prediction": TARGET_NAMES[pred_idx],
        "confidence": round(max(probs), 4),
        "probabilities": {
            TARGET_NAMES[i]: round(p, 4) for i, p in enumerate(probs)
        },
    }


@app.get("/")
def root():
    return {
        "message": "Iris Classifier API is running",
        "docs": "/docs",
        "predict_endpoint": "/predict (POST)",
    }
