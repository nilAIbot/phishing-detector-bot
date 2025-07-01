from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from src.models.phishsleuth_text import PhishSleuthText

app = FastAPI()

# Model loading
BACKBONE = "bert-base-uncased"
MODEL_PATH = "./artifacts/model.pt"  # Adjusted path


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = PhishSleuthText(BACKBONE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: TextRequest):
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


@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "PhishSleuth API is running"}