'''
Trains a wav2vec2-based classifier to detect synthetic data in the dataset 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class w2v2_clf(nn.Module):
    def __init__(self, freeze_encoder=True, hidden_dim=256):
        super().__init__()
        
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        encoder_dim = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(2 * encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, x):
        out = self.encoder(
            input_values=x,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state

        mean = hidden.mean(dim=1)
        std = hidden.std(dim=1)

        pooled = torch.cat([mean, std], dim=-1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
