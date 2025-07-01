import json
import torch
import yaml
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from models.phishsleuth_text import PhishSleuthText
from utils.logger import get_logger


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        url = " ".join(r.get("urls", [])[:1])
        txt = f"{r['text']} [URL] {url}"
        enc = self.tok(txt,
                       truncation=True,
                       padding="max_length",
                       max_length=self.max_len,
                       return_tensors="pt",
                       return_token_type_ids=False)
        y = 1 if r['label'] == 'phish' else 0
        return {k: v.squeeze(0) for k, v in enc.items()}, y


def main(cfg):
    log = get_logger()

    # Set device (automatically detects CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Ensure numeric values are properly typed
    cfg['train']['lr'] = float(cfg['train']['lr'])
    cfg['train']['weight_decay'] = float(cfg['train']['weight_decay'])
    cfg['model']['max_len'] = int(cfg['model']['max_len'])
    cfg['train']['batch_size'] = int(cfg['train']['batch_size'])
    cfg['train']['epochs'] = int(cfg['train']['epochs'])

    # Create output directory if it doesn't exist
    model_out_dir = os.path.dirname(cfg['paths']['model_out'])
    os.makedirs(model_out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg['model']['backbone'])
    train_ds = JsonlDataset(cfg['paths']['processed'] + '/train.jsonl', tok, cfg['model']['max_len'])
    val_ds = JsonlDataset(cfg['paths']['processed'] + '/val.jsonl', tok, cfg['model']['max_len'])

    train_dl = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = PhishSleuthText(cfg['model']['backbone']).to(device)
    opt = AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    sched = get_linear_schedule_with_warmup(opt, 0, len(train_dl) * cfg['train']['epochs'])
    criterion = torch.nn.CrossEntropyLoss()

    for ep in range(cfg['train']['epochs']):
        model.train()
        total = 0
        correct = 0

        for batch, y in train_dl:
            # Move data to the same device as model
            b = {k: v.to(device) for k, v in batch.items()}
            y = y.to(device)

            # Forward pass
            outputs = model(**b)
            loss = criterion(outputs, y)

            # Backward pass and optimize
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # Save model after each epoch
        torch.save(model.state_dict(), cfg['paths']['model_out'])
        log.info(f"Epoch {ep + 1}: Saved model → {cfg['paths']['model_out']}")
        log.info(f"Training Accuracy: {100 * correct / total:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch, y in val_dl:
                b = {k: v.to(device) for k, v in batch.items()}
                y = y.to(device)
                outputs = model(**b)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        log.info(f"Validation Accuracy: {100 * val_correct / val_total:.2f}%")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("./../configs/default.yaml"))
    main(cfg)