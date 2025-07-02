import gradio as gr
import torch
from transformers import AutoTokenizer
from src.models.phishsleuth_text import PhishSleuthText
import gdown
import os

# ✅ Configuration
BACKBONE = "bert-base-uncased"
MODEL_URL = "https://drive.google.com/uc?id=1pchHLgg6m4FvMfpSrkJkNBuqnbo6mmB1"
MODEL_PATH = "./artifacts/model.pt"

# ✅ Ensure model exists
os.makedirs("./artifacts", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model download failed")

# ✅ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
model = PhishSleuthText(BACKBONE)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded successfully")

# ✅ Inference function
def classify_phishing(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        logits = model(**inputs)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()

    result = {
        "Text": text,
        "Probability (Phishing)": f"{prob:.2%}",
        "Is Phishing?": prob > 0.5,
        "Message": "🚫 Don't click!" if prob > 0.5 else "✅ Safe to proceed"
    }
    return result

# ✅ Gradio Interface
demo = gr.Interface(
    fn=classify_phishing,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Paste suspicious email or link text here..."
    ),
    outputs="json",
    title="PhishSleuth 🔍",
    description="Detects phishing likelihood using a fine-tuned BERT model."
)

# ✅ Run
if __name__ == "__main__":
    demo.launch()
