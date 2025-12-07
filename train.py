import random
import yaml
import inspect

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf

from layers import DirtModel, KVCache


class ParquetDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.ds = dataframe

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds.iloc[idx]
        return item['text'], torch.tensor(item['tokens'])   # or torch.from_numpy


def collate_fn(batch, config, do_augment, device):
    texts, tokens = zip(*batch)
    
    # text
    
    max_text = config.max_text_len
    pad_tok = config.text_pad_id
    text_ids = []
    for i, txt in enumerate(texts):
        if do_augment:
            r = random.random()
            if r < 0.15:
                txt = txt.lower()
            elif r < 0.3:
                #txt = norm_texts[i]
                txt = txt.translate(str.maketrans('', '', '.,!?;"\''))

        b_full = txt.encode('utf-8')
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids).to(device)
    src_pos = torch.arange(max_text).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)
    
    # audio

    max_audio = config.max_audio_len
    seq_lens = [min(e.size(0), max_audio) for e in tokens]
    batch_max = max(seq_lens)
    padded = [F.pad(e, (0, batch_max - e.size(0))) if e.size(0) < batch_max else e[:batch_max] for e in tokens]
    codes = torch.stack(padded).to(device)  # (B, T=batch_max)
    B, T = codes.shape

    max_tgt_len = max_audio + 2
    pad_val = config.audio_pad_id
    bos_val = config.audio_bos_id
    eos_val = config.audio_eos_id
    tgt = torch.full((B, max_tgt_len), pad_val, dtype=torch.long, device=device)
    tgt[:, 0] = bos_val
    tgt_lens = []
    for i, L in enumerate(seq_lens):
        tgt[i, 1:1+L] = codes[i, :L]
        tgt[i, 1+L] = eos_val
        tgt_lens.append(1+L+1)
    tgt = tgt.unsqueeze(-1) # add an extra dim for channels for compatibility
    
    tgt_pos = torch.arange(max_tgt_len, device=device).unsqueeze(0).expand(B, -1)
    tgt_pad = tgt.ne(pad_val).any(-1)

    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len),
                                    dtype=torch.bool,
                                    device=device))
    dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
    dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    return {
        'src_tokens': src,
        'src_positions': src_pos.to(device),
        'enc_self_attn_mask': enc_self_attn_mask,
        'tgt_tokens': tgt,
        'tgt_positions': tgt_pos.to(device),
        'dec_self_attn_mask': dec_self_attn_mask,
        'dec_cross_attn_mask': dec_cross_attn_mask,
        'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long, device=device),
    }


def configure_optimizers(model, config, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=config.learning_rate, 
        betas=(config.beta1, config.beta2),
        eps=config.adamw_eps, #
        **extra_args
    )
    print(f"using fused AdamW: {use_fused}")
    return optimizer


def eval_step(model, val_dataloader, config):
    eval_losses = []
    with torch.inference_mode():
        for i, eval_batch in enumerate(val_dataloader):
            logits = model(
                src_ids=eval_batch['src_tokens'],
                src_pos=eval_batch['src_positions'],
                tgt_ids=eval_batch['tgt_tokens'],
                tgt_pos=eval_batch['tgt_positions'],
                enc_sa_mask=eval_batch['enc_self_attn_mask'],
                dec_sa_mask=eval_batch['dec_self_attn_mask'],
                dec_ca_mask=eval_batch['dec_cross_attn_mask'],
            )[:, :-1]
            target = eval_batch['tgt_tokens'][:, 1:]
            V_e = logits.size(-1)
            lc = logits.reshape(-1, V_e)
            tc = target.reshape(-1)
            eval_loss = F.cross_entropy(lc, tc, ignore_index=config.model.audio_pad_id)
            eval_losses.append(eval_loss)
    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    return avg_eval_loss.item()


def train_step(model, batch, optimizer, scheduler, config):
    if random.random() < config.training.unconditional_frac:
        pad_tok = config.model.text_pad_id
        batch['src_tokens'] = torch.zeros_like(batch['src_tokens'])
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(
            src_ids=batch['src_tokens'],
            src_pos=batch['src_positions'],
            tgt_ids=batch['tgt_tokens'],
            tgt_pos=batch['tgt_positions'],
            enc_sa_mask=batch['enc_self_attn_mask'],
            dec_sa_mask=batch['dec_self_attn_mask'],
            dec_ca_mask=batch['dec_cross_attn_mask'],
        ) # (B, T, V)

    # cut the length
    lens = batch['tgt_lens']
    max_L = int(lens.max().item())
    logits = logits[:, :max_L-1]
    target = batch['tgt_tokens'][:, 1:max_L, :]

    # mask padding and then calculate loss
    B, Tm1, C = target.shape
    time_idx = torch.arange(Tm1).unsqueeze(0).to(target.device)
    valid_time = time_idx < (lens.unsqueeze(1) - 1)
    mask = valid_time.unsqueeze(-1).expand(-1, -1, C)

    _, T, V = logits.size()
    lc = logits.reshape(-1, V)
    tc = target.reshape(-1)
    mc = mask.reshape(-1)
    lc_valid = lc[mc]
    tc_valid = tc[mc]
    loss_ce = F.cross_entropy(lc_valid, tc_valid, ignore_index=config.model.audio_pad_id)

    # timing loss
    eos_probs = F.softmax(logits[..., config.model.audio_eos_id], dim=-1)  # (B, T)
    positions = torch.arange(T, device=logits.device).float()  # (1, T)
    expected_eos_pos = (eos_probs * positions.unsqueeze(0)).sum(dim=-1)
    target_eos_pos = lens.float() - 1  # (B,)
    diff = expected_eos_pos - target_eos_pos
    loss_timing = (diff.clamp(min=0) ** 2).mean()
    #loss_timing = F.mse_loss(expected_eos_pos, target_eos_pos, reduction='mean')

    # total loss
    loss = loss_ce + 5e-4 * loss_timing
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return loss, loss_ce, loss_timing, grad_norm


def main():
    config = OmegaConf.load('config.yaml')

    device = config.training.device
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Prepare data
    df = pd.read_parquet(config.training.data_path)
    train_ds, val_ds = train_test_split(df, test_size=16, random_state=42)
    train_dataloader = DataLoader(
        ParquetDataset(train_ds),
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, config.model, True, device),
    )
    val_dataloader = DataLoader(
        ParquetDataset(val_ds),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, config.model, False, device),
    )
    print(f"train: {len(train_dataloader)} batches\nval: {len(val_dataloader)} batches")

    # Initialize model
    model = DirtModel(**config.model.architecture).to(device)
    if config.training.torch_compile:
        #model = torch.compile(model, mode='max-autotune')
        model = torch.compile(model)
        print("Using torch.compile() for model.")

    optimizer = configure_optimizers(model, config.training, device_type='cuda')
    scheduler = get_scheduler(
        config.training.lr_decay_style, optimizer,
        num_warmup_steps=config.training.warmup_steps, num_training_steps=config.training.max_steps
    )

    # train
    if config.training.wandb.use_wandb:
        import wandb
        wandb.init(
            project=config.training.wandb.project_name,
            name=config.training.wandb.run_name,
            config=OmegaConf.to_container(config, resolve=True),
        )
        wandb.watch(model, log="all")

    model.train()

    step = 0
    epoch = 0
    while True:
        train_iterloader = iter(train_dataloader)
        for batch in train_iterloader:
            loss, loss_ce, loss_timing, grad_norm = train_step(model, batch, optimizer, scheduler, config)
            
            # logging
            current_lr = scheduler.get_last_lr()[0]
            if step % config.training.eval_interval == 0:
                eval_loss = eval_step(model, val_dataloader, config)
                print("eval ", step, current_lr, loss.item(), eval_loss)
                if config.training.wandb.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "eval/loss": eval_loss,
                        "grad_norm": grad_norm,
                        "lr": current_lr,
                        "step": step,
                    })
            elif step % config.training.log_interval == 0:
                print("train ", step, current_lr, loss.item())
                if config.training.wandb.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/ce_loss": loss_ce.item(),
                        "train/timing_loss": loss_timing.item(),
                        "grad_norm": grad_norm,
                        "lr": current_lr,
                        "step": step,
                    })
                
            # checkpoint
            if step > 0 and step % config.training.save_interval == 0:
                ckpt = f"dirt_{step}.pth"
                opt_ckpt = f"optim_{step}.pth"
                torch.save(model.state_dict(), ckpt)
                torch.save(optimizer.state_dict(), opt_ckpt)
            
            step += 1
        if step >= config.training.max_steps: break

if __name__ == "__main__":
    main()