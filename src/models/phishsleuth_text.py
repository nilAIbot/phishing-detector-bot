import torch.nn as nn
from transformers import AutoModel

class PhishSleuthText(nn.Module):
    def __init__(self, backbone="bert-base-uncased", num_labels=2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone)
        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(self.enc.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.enc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs  # passes token_type_ids if present
        )
        hidden = outputs.last_hidden_state[:, 0]
        return self.cls(self.dropout(hidden))