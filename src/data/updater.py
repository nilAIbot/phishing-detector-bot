"""Run collector → preprocess → merge into train/val/test splits."""
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

from collector import update as collect
from preprocess import process_nazario  # add others if needed

RAW_DIR = Path("./../../data/raw")
PROC_DIR = Path("../../data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def build_dataset():
    # Step 1: Download latest raw files
    collect()

    # Step 2: Preprocess raw phishing files
    samples = []
    for gz in RAW_DIR.glob("enron_*.tar.gz"):
        try:
            samples.extend(process_nazario(gz))
        except Exception as e:
            print(f"[ERROR] Failed to process {gz.name} → {e}")

    # TODO: Add Enron benign emails here (e.g., process_enron)

    # Step 3: Split into train/val/test
    if not samples:
        print("❌ No samples found. Check downloads or preprocessing.")
        return

    random.shuffle(samples)
    train, tmp = train_test_split(samples, test_size=0.2, random_state=42)
    val, test = train_test_split(tmp, test_size=0.5, random_state=42)

    # Step 4: Save splits
    for name, split in zip(["train", "val", "test"], [train, val, test]):
        out = PROC_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row) + "\n")
        print(f"✅ Saved {len(split)} samples → {out}")

if __name__ == "__main__":
    build_dataset()
