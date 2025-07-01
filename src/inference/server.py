from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from src.models.phishsleuth_text import PhishSleuthText
import gdown
import os
from typing import Optional

app = FastAPI()

# Configuration
BACKBONE = "bert-base-uncased"
MODEL_URL = "https://drive.google.com/uc?id=1pchHLgg6m4FvMfpSrkJkNBuqnbo6mmB1"
MODEL_PATH = "./artifacts/model.pt"

# Global variables for model and tokenizer
tokenizer: Optional[AutoTokenizer] = None
model: Optional[PhishSleuthText] = None


def download_model():
    """Download model from Google Drive if not exists"""
    try:
        os.makedirs("./artifacts", exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            print("Downloading model from Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            if not os.path.exists(MODEL_PATH):
                raise RuntimeError("Model download failed")
    except Exception as e:
        print(f"Model download error: {str(e)}")
        raise


def load_model():
    """Load tokenizer and model if not already loaded"""
    global tokenizer, model

    if tokenizer is None or model is None:
        download_model()  # Ensure model is downloaded
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
        model = PhishSleuthText(BACKBONE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        print("Model loaded successfully")


@app.on_event("startup")
async def startup_event():
    """Initialize model when application starts"""
    try:
        load_model()
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: TextRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            logits = model(**inputs)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()

        return {
            "text": request.text,
            "is_phishing": prob > 0.5,
            "probability": prob,
            "message": "Don't click" if prob > 0.5 else "Safe to proceed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
    status = {
        "status": "healthy" if tokenizer and model else "unhealthy",
        "message": "PhishSleuth API is running",
        "model_loaded": bool(tokenizer and model)
    }
    return status