"""Quick evaluation script for trained model on val/test splits."""
import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.phishsleuth_text import PhishSleuthText
from utils.metrics import compute_metrics

# ────────────────────────────────────────────────────────────────────────────
class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        url = " ".join(r.get("urls", [])[:1])
        txt = f"{r['text']} [URL] {url}"
        enc = self.tok(
            txt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False
        )
        y = 1 if r["label"] == "phish" else 0
        return {k: v.squeeze(0) for k, v in enc.items()}, y

# ────────────────────────────────────────────────────────────────────────────
def format_metric_value(value):
    """Helper function to format metric values that might be nan or lists"""
    if isinstance(value, (list, np.ndarray)):
        return [f"{x:.4f}" if isinstance(x, (float, int)) else str(x) for x in value]
    elif isinstance(value, (float, int)):
        return f"{value:.4f}"
    return str(value)

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--cfg", default="./../configs/default.yaml")
    args = ap.parse_args()

    # Load configuration
    cfg = yaml.safe_load(open(args.cfg))

    # Initialize tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])

    # Load dataset
    ds_path = Path(cfg["paths"]["processed"]) / f"{args.split}.jsonl"
    ds = JsonlDataset(ds_path, tok, cfg["model"]["max_len"])
    dl = DataLoader(ds, batch_size=32)

    # Device selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Initialize and load model
    model = PhishSleuthText(cfg["model"]["backbone"]).to(DEVICE)
    model.load_state_dict(torch.load(cfg["paths"]["model_out"], map_location=DEVICE))
    model.eval()

    # Evaluation loop
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch, y in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            y_prob.extend(probs)
            y_true.extend(y)

    # Compute metrics
    metrics = compute_metrics(y_true, y_prob)

    # Print results
    print(f"\nEvaluation results on {args.split} set:")
    print("-" * 40)
    for k, v in metrics.items():
        formatted_value = format_metric_value(v)
        print(f"{k:<15}: {formatted_value}")
    print("-" * 40)

    # Check for class imbalance
    unique_classes = set(y_true)
    if len(unique_classes) == 1:
        print("\nWarning: Only one class present in y_true!")
        print(f"Class present: {'phish' if list(unique_classes)[0] == 1 else 'benign'}")