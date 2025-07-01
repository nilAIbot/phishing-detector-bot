"""Incrementally fine‑tune on new data each night."""
import datetime as dt
import json
import torch
import yaml
from models.phishsleuth_text import PhishSleuthText
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW

CFG = yaml.safe_load(open("configs/default.yaml"))
TOK = AutoTokenizer.from_pretrained(CFG['model']['backbone'])

class DayDataset(torch.utils.data.Dataset):
    def __init__(self, date_str):
        path = f"data/processed/daily/{date_str}.jsonl"
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        txt = f"{r['text']} [URL] {' '.join(r.get('urls',[])[:1])}"
        enc = TOK(txt, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        y = 1 if r['label']=='phish' else 0
        return {k:v.squeeze(0) for k,v in enc.items()}, y

def incremental_update(date_str):
    ds = DayDataset(date_str)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    model = PhishSleuthText(CFG['model']['backbone'])
    model.load_state_dict(torch.load(CFG['paths']['model_out']))
    model.train()
    opt = AdamW(model.parameters(), lr=1e-6)
    for batch, y in dl:
        b = {k:v for k,v in batch.items()}
        loss = torch.nn.functional.cross_entropy(model(**b), torch.tensor(y))
        loss.backward(); opt.step(); opt.zero_grad()
    torch.save(model.state_dict(), CFG['paths']['model_out'])
    print(f"✅ model updated with {len(ds)} new samples")

if __name__ == "__main__":
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    incremental_update(yesterday)