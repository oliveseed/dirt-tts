import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import ipdb
device = 'cuda'
torch.set_float32_matmul_precision('high')


class ParquetDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.ds = dataframe
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds.iloc[idx]
        return item['data'], item['label']


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


def collate_fn(batch):
    mp3_data, labels = zip(*batch)
    waveforms = []
    for data in mp3_data:
        waveform, sr = torchaudio.load(data)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        num_samples = waveform.shape[1]
        target_len = 3*16000
        if num_samples >= target_len:
            start = random.randint(0, num_samples-target_len)
            waveform = waveform[:, start:start+target_len]
        else:
            pad_len = target_len-num_samples
            waveform = F.pad(waveform, (0, pad_len))

        waveforms.append(waveform.squeeze(0))
    return torch.stack(waveforms), torch.tensor(labels)


def eval_step(model, val_dataloader):
    eval_losses = []
    eval_preds = []
    model.eval()
    with torch.inference_mode():
        for eval_batch in val_dataloader:
            waveforms, labels = eval_batch[0].to(device), eval_batch[1].to(device)
            logits = model(waveforms)

            eval_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            eval_losses.append(eval_loss)

            preds = torch.sigmoid(logits)
            pred_class = (preds > 0.5).long()
            eval_preds.extend(zip(labels.cpu().numpy(), pred_class.cpu().numpy()))
    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    y_true, y_pred = zip(*eval_preds)
    report = classification_report(y_true, y_pred)#, output_dict=True)
    print(report)
    for i in range(len(labels)):
        print(labels[i].item(), preds[i].item())
    model.train()
    return avg_eval_loss.item()


if __name__ == "__main__":
    ds = pd.read_parquet("data/emilia_synth_labels.parquet")
    ds_spkrs = ds['speaker_id'].unique()
    train_spkrs, val_spkrs = train_test_split(ds_spkrs, test_size=32, random_state=4321)
    train_ds = ds[ds['speaker_id'].isin(train_spkrs)]
    val_ds = ds[ds['speaker_id'].isin(val_spkrs)]
    train_dataloader = DataLoader(
        ParquetDataset(train_ds),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        ParquetDataset(val_ds),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"train: {len(train_dataloader)} batches\nval: {len(val_dataloader)} batches")

    model = w2v2_clf(freeze_encoder=True, hidden_dim=3072)
    model = model.to(device)

    # dummy_input = torch.randn(2, 16000)  # Batch of 2 samples, each 1 second at 16kHz
    # output = model(dummy_input)
    # print(output.shape)  # Should print: torch.Size([2])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_scheduler(
        'linear', optimizer,
        num_warmup_steps=0, num_training_steps=10000
    )

    model.train()

    step = 0
    epoch = 0
    while True:
        train_iterloader = iter(train_dataloader)
        for batch in train_iterloader:
            waveforms, labels = batch[0].to(device), batch[1].to(device)

            logits = model(waveforms)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0]
            if step % 200 == 0:
                eval_loss = eval_step(model, val_dataloader)
                print(f"Step {step} | Eval Loss: {eval_loss:.4f}")
            elif step % 10 == 0:
                print(f"Step {step} | LR: {current_lr:.6f} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f}")

            step += 1
        epoch += 1
        print(f"Completed epoch {epoch}")
        break

    torch.save(model.classifier.state_dict(), f"w2v2_clf_head_{step}.pth")