'''
Single GPU training script

Usage:
    python3 train.py
'''

import time
import re
import inspect
import random
import yaml
import wandb
import ipdb
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast

from transformers import get_scheduler#, AutoTokenizer
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

from model import TTS


device = 'cuda'
torch.set_float32_matmul_precision('high')


_last_log_time = None
_last_log_step = None


class TrainingConfig:
    # optimization -------------------------------------
    max_steps:                  int = 300010
    num_epochs:                 int = 2
    encoder_learning_rate:      float = 5e-5
    decoder_learning_rate:      float = 1e-4
    batch_size:                 int = 32
    grad_accum_steps:           int = 2
    weight_decay:               float = 0.1
    beta1:                      float = 0.9
    beta2:                      float = 0.999
    adamw_eps:                  float = 1e-8
    # sometimes zero encoder input for cfg -------------
    unconditional_frac:         float = 0.15
    # size of validation dataset ----------------------- 
    val_split_size:             int = 16
    # max seqlens to train on --------------------------
    text_maxlen:                int = 128
    audio_maxlen:               int = 768   # 30 secs
    # special tokens -----------------------------------
    text_eos_value:             int = 2     # llama
    audio_eos_value:            int = 46657
    audio_bos_value:            int = 46658
    # logging and checkpoint ---------------------------
    use_wandb:                  bool = False
    wandb_project_name:         str = "nanotts"
    train_log_interval:         int = 100
    eval_log_interval:          int = 10000
    ckpt_interval:              int = 20000
    # path ---------------------------------------------
    dataset_path:               str = '../data/emilia_300_clean.npy'


def augment_text(text: str) -> str:
    # Only apply any augmentation 30% of the time
    if random.random() >= 0.3:
        return text

    # Case normalization
    r = random.random()
    if r < 0.15:
        text = text.lower()
    elif r < 0.30:
        text = text.upper()

    # Remove periods and commas ONLY if followed by space or end of string
    if random.random() < 0.25:
        text = re.sub(r'[.,](?=\s|$)', '', text)

    # Replace some periods (only if followed by space) with newline
    # Also remove the space after the period
    if random.random() < 0.30:
        def replace_period(match):
            # match is ". "
            return '\n' if random.random() < 0.5 else match.group(0)
        
        text = re.sub(r'\. ', replace_period, text)

    return text


class ParquetDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.ds = dataframe#.values
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        return item[0], item[6]


def collate_fn(batch, config, tokenizer, do_augment, device):
    texts, audio_tokens = zip(*batch)
    
    # --------------------- text ---------------------
    max_text = config.text_maxlen
    text_eos = config.text_eos_value

    src = []
    src_pos = []
    src_lens = []
    src_cu_lens = [0]

    pos = 0
    for i, txt in enumerate(texts):
        if do_augment:  # training only
            txt = augment_text(txt)
        
        txt_ids = tokenizer(txt, add_special_tokens=False)['input_ids'][:max_text-1]
        txt_ids = list(txt_ids) + [text_eos]
        n = len(txt_ids)

        pos += n
        src.extend(txt_ids)
        src_lens.append(n)
        src_pos.extend(np.arange(n, dtype=np.int64))
        src_cu_lens.append(pos)

    src = torch.tensor(src)
    src_pos = torch.tensor(src_pos)
    src_cu_lens = torch.tensor(src_cu_lens, dtype=torch.int32)
    src_max_seqlen = max(src_lens)

    # --------------------- audio ---------------------
    max_audio = config.audio_maxlen
    audio_bos = config.audio_bos_value
    audio_eos = config.audio_eos_value
    
    audio_total = sum(map(len, audio_tokens)) + 2 * len(audio_tokens)
    tgt = np.empty(audio_total, dtype=np.int64)
    tgt_pos = np.empty(audio_total, dtype=np.int64)
    tgt_cu_lens = torch.zeros(len(audio_tokens)+1, dtype=torch.int32)

    pos = 0
    for i, audio_seq in enumerate(audio_tokens):
        audio_seq = audio_seq[:max_audio-2]
        n = audio_seq.size
        tgt[pos] = audio_bos
        tgt[pos+1:pos+1+n] = audio_seq
        tgt[pos+n+1] = audio_eos
        tgt_pos[pos:pos+n+2] = np.arange(n+2, dtype=np.int64)
        pos += n + 2
        tgt_cu_lens[i+1] = pos

    tgt = torch.from_numpy(tgt)
    tgt_pos = torch.from_numpy(tgt_pos)
    tgt_max_seqlen = max([len(seq)+2 for seq in audio_tokens])

    return {
        'src_tokens': src.to(device),
        'src_positions': src_pos.to(device),
        'src_cu_seqlens': src_cu_lens.to(device),
        'src_max_seqlen': src_max_seqlen,
        'tgt_tokens': tgt.to(device),
        'tgt_positions': tgt_pos.to(device),
        'tgt_cu_seqlens': tgt_cu_lens.to(device),
        'tgt_max_seqlen': tgt_max_seqlen,
    }


@torch.inference_mode()
def eval_step(tts, val_dataloader):
    
    eval_losses = []
    eval_similarities = []

    for i, eval_batch in enumerate(val_dataloader):
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            eval_loss = tts.model(
                src_ids=eval_batch['src_tokens'],
                src_pos=eval_batch['src_positions'],
                src_cu_seqlens=eval_batch['src_cu_seqlens'],
                src_max_seqlen=eval_batch['src_max_seqlen'],
                tgt_ids=eval_batch['tgt_tokens'],
                tgt_pos=eval_batch['tgt_positions'],
                tgt_cu_seqlens=eval_batch['tgt_cu_seqlens'],
                tgt_max_seqlen=eval_batch['tgt_max_seqlen'],
            )
        eval_losses.append(eval_loss.item())

        # generate outputs from the text prompts in this batch
        # not batched because that would require all sequences same length
        for j in range(config.val_split_size):
            src_start, src_end = eval_batch['src_cu_seqlens'][j], eval_batch['src_cu_seqlens'][1+j]
            src_slice = eval_batch['src_tokens'][src_start:src_end]

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                gen_codes, gen_lens = tts.generate(src_slice)
            print(gen_lens)
            if not (gen_codes.ndim == 3 and gen_lens[0].item() > 0):
                print("Generated an undecodable sequence!", gen_codes)
                continue
            # exclude bos and eos tokens from waveform decoding
            codes = gen_codes[0, 1:-1, :]
            
            decoded_pred_audio, decoded_pred_np = tts.decode(
                codes=codes,
                device=device
            )

            wavfile.write(f'eval_out/eval_output_{config.batch_size*i+j}.wav', 16000, decoded_pred_np)

    avg_eval_loss = sum(eval_losses) / len(eval_losses)

    print("eval ", step, avg_eval_loss)#, avg_eval_sim)
    if config.use_wandb:
        wandb.log({
            "eval_loss": avg_eval_loss,
            "step": step / config.grad_accum_steps,
        })


def train_step(model, batch, optimizer, scheduler, config, step):
    global _last_log_time, _last_log_step
    if config.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    if random.random() < config.unconditional_frac:
        batch['src_tokens'] = torch.zeros_like(batch['src_tokens'])

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(
            src_ids=batch['src_tokens'],
            src_pos=batch['src_positions'],
            src_cu_seqlens=batch['src_cu_seqlens'],
            src_max_seqlen=batch['src_max_seqlen'],
            tgt_ids=batch['tgt_tokens'],
            tgt_pos=batch['tgt_positions'],
            tgt_cu_seqlens=batch['tgt_cu_seqlens'],
            tgt_max_seqlen=batch['tgt_max_seqlen'],
        )

    (loss / config.grad_accum_steps).backward()

    should_step = (step + 1) % config.grad_accum_steps == 0
    if should_step:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        effective_step = (step + 1) // config.grad_accum_steps
        if effective_step % config.train_log_interval == 0:
            now = time.time()
            speed = None
            if _last_log_time is not None and _last_log_step is not None:
                elapsed = now - _last_log_time
                steps = effective_step - _last_log_step
                if elapsed > 0 and steps > 0:
                    speed = steps / elapsed
            _last_log_time = now
            _last_log_step = effective_step

            current_lrs = scheduler.get_last_lr()
            enc_lr, dec_lr = current_lrs[0], current_lrs[2]

            train_loss = loss.item()
            
            encoder_params = list(model.encoder.parameters())
            decoder_params = list(model.decoder.parameters())
            encoder_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in encoder_params if p.grad is not None]), 2).item() if encoder_params else 0.0
            decoder_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in decoder_params if p.grad is not None]), 2).item() if decoder_params else 0.0

            # Print grad norms
            grad_str = f"train {effective_step} ({step}) | {dec_lr:.4f} | {train_loss:.4f}"
            if speed is not None:
                grad_str += f" | {speed:.2f} it/s"
            grad_str += f" | encoder_gn: {encoder_grad_norm:.4f} | decoder_gn: {decoder_grad_norm:.4f}"
            print(grad_str)

            if config.use_wandb:
                log_dict = {
                    "train_loss": train_loss,
                    "grad_norm/grad_norm": grad_norm,
                    "grad_norm/encoder_grad_norm": encoder_grad_norm,
                    "grad_norm/decoder_grad_norm": decoder_grad_norm,
                    "lr/encoder_lr": enc_lr,
                    "lr/decoder_lr": dec_lr,
                    "step": effective_step,
                }
                if speed is not None:
                    log_dict["train_speed_it_per_sec"] = speed
                wandb.log(log_dict)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)


def configure_optimizers(model, config, device_type):
    param_dict = {
        pn: p for pn, p in model.named_parameters()
        if p.requires_grad
    }

    # Separate encoder / decoder params
    encoder_params = {
        n: p for n, p in param_dict.items()
        if n.startswith("encoder.") or n.startswith("_orig_mod.encoder.")
    }
    decoder_params = {
        n: p for n, p in param_dict.items()
        if n.startswith("decoder.") or n.startswith("_orig_mod.decoder.")
    }

    # Weight decay split
    enc_decay = [p for n, p in encoder_params.items() if p.dim() >= 2]
    enc_nodecay = [p for n, p in encoder_params.items() if p.dim() < 2]

    dec_decay = [p for n, p in decoder_params.items() if p.dim() >= 2]
    dec_nodecay = [p for n, p in decoder_params.items() if p.dim() < 2]

    optim_groups = [
        {
            "params": enc_decay,
            "weight_decay": config.weight_decay,
            "lr": config.encoder_learning_rate,
        },
        {
            "params": enc_nodecay,
            "weight_decay": 0.0,
            "lr": config.encoder_learning_rate,
        },
        {
            "params": dec_decay,
            "weight_decay": config.weight_decay,
            "lr": config.decoder_learning_rate,
        },
        {
            "params": dec_nodecay,
            "weight_decay": 0.0,
            "lr": config.decoder_learning_rate,
        },
    ]

    # Stats
    print(
        f"encoder decay tensors: {len(enc_decay)}, "
        f"params: {sum(p.numel() for p in enc_decay):,}"
    )
    print(
        f"encoder no-decay tensors: {len(enc_nodecay)}, "
        f"params: {sum(p.numel() for p in enc_nodecay):,}"
    )
    print(
        f"decoder decay tensors: {len(dec_decay)}, "
        f"params: {sum(p.numel() for p in dec_decay):,}"
    )
    print(
        f"decoder no-decay tensors: {len(dec_nodecay)}, "
        f"params: {sum(p.numel() for p in dec_nodecay):,}"
    )

    fused_available = (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
    )
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}

    optimizer = torch.optim.AdamW(
        optim_groups,
        betas=(config.beta1, config.beta2),
        eps=config.adamw_eps,
        **extra_args,
    )

    print(f"using fused AdamW: {use_fused}")

    return optimizer


if __name__ == "__main__":
    config = TrainingConfig()

    # 1) prepare model
    with open('config.yaml', 'r') as f:
        dirt_config = yaml.safe_load(f)
    tts = TTS(dirt_config)
    
    model = tts.model
    model.encoder.embedding.requires_grad_(False)
    model._init_weights()
    model = model.to(device)
    model = torch.compile(model) # should be fullgraph, but don't explicitly set it due to dynamic shapes

    tokenizer = tts.text_tokenizer


    # 2) prepare optimizer
    optimizer = configure_optimizers(model, config, device_type=device)
    scheduler = get_scheduler(
        'linear', optimizer,
        num_warmup_steps=0, num_training_steps=config.max_steps
    )


    # 3) prepare data
    data = np.load(config.dataset_path, allow_pickle=True)
    train_ds, val_ds = train_test_split(data, test_size=config.val_split_size, random_state=42)

    train_dataloader = torch.utils.data.DataLoader(
        ParquetDataset(train_ds),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, config, tokenizer, True, device),
    )
    val_dataloader = torch.utils.data.DataLoader(
        ParquetDataset(val_ds),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, config, tokenizer, False, device),
    )

    print(f"train: {len(train_dataloader)} batches ({len(train_dataloader) / config.grad_accum_steps} effective batches)\nval: {len(val_dataloader)} batches")


    # 4) train
    if config.use_wandb:
        wandb.init(project=config.wandb_project_name)

    model.train()
    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            train_step(model, batch, optimizer, scheduler, config, step)
            
            if (step / config.grad_accum_steps) % config.eval_log_interval == 0:
                eval_step(tts, val_dataloader)
            
            # checkpoint
            # if step > 0 and step % config.ckpt_interval == 0:

        # torch.save(model.state_dict(), f"frampton3_1_4e12d_2_epoch{epoch}.pth")

    # torch.save(optimizer.state_dict(), f"optim.pth")