import os, sys, time, math, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# os.chdir('./')
sys.path.insert(0, os.getcwd())
from layers3_1 import TTSModel
from train_frampton_3_1 import TrainingConfig, ParquetDataset, collate_fn, configure_optimizers
from transformers import get_scheduler
import yaml

# Keep the run deterministic enough for repeatability.
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.set_float32_matmul_precision('high')

device = 'cuda'
config = TrainingConfig()
config.use_wandb = False

with open('config.yaml', 'r') as f:
    tts_config = yaml.safe_load(f)
model_cfg = tts_config['model']['architecture']
model = TTSModel(**model_cfg)
model.encoder.embedding.requires_grad_(False)
model = model.to(device)
model = torch.compile(model)
optimizer = configure_optimizers(model, config, device_type=device)
scheduler = get_scheduler('constant', optimizer, num_warmup_steps=5000, num_training_steps=config.max_steps)

tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
data = np.load('../data/emilia_300_clean.npy', allow_pickle=True)
train_ds, _ = train_test_split(data, test_size=16, random_state=42)
loader = DataLoader(
    ParquetDataset(train_ds),
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=lambda x: collate_fn(x, config, tokenizer, True, device),
)

# Pre-collate a short stable window so measurement excludes CPU data loading/tokenization.
batches = []
for i, batch in enumerate(loader):
    batches.append(batch)
    if len(batches) >= 8:
        break

def lens_from_cu(cu):
    return (cu[1:] - cu[:-1]).detach().cpu().numpy().astype(np.int64)

def flops_for_batch(batch):
    # Dense model FLOPs for the exact packed token counts in this batch.
    # Count matmul/linear as 2 FLOPs per multiply-add. Training fwd+bwd = 3x fwd matmul work.
    # Attention counts QK^T and AV matmuls, also fwd+bwd = 3x fwd attention matmul work.
    cfg = model_cfg
    src_lens = lens_from_cu(batch['src_cu_seqlens'])
    tgt_lens = lens_from_cu(batch['tgt_cu_seqlens'])
    S = int(src_lens.sum())
    T = int(tgt_lens.sum())
    src_sq = int((src_lens * src_lens).sum())
    tgt_causal_pairs = int((tgt_lens * (tgt_lens + 1) // 2).sum())
    cross_pairs = int((src_lens * tgt_lens).sum())

    enc_layers = cfg['enc_num_layers']
    dec_layers = cfg['dec_num_layers']
    Eenc = cfg['enc_embed_dim']
    Henc = cfg['enc_hidden_dim']
    Edec = cfg['dec_embed_dim']
    Hdec = cfg['dec_hidden_dim']
    nh = cfg['num_query_heads']
    h_enc = cfg['enc_head_dim']
    h_dec = cfg['dec_head_dim']
    ca_nh = cfg['ca_num_query_heads']
    ca_h = cfg['ca_head_dim']
    V = cfg['dec_vocab_size']

    fwd = 0
    # Encoder layer: fused qkv, out proj, MLP in/out, full self-attn matmuls.
    enc_linear = 2*S*Eenc*(3*Eenc) + 2*S*Eenc*Eenc + 2*S*Eenc*Henc + 2*S*Henc*Eenc
    enc_attn = 4*nh*h_enc*src_sq
    fwd += enc_layers * (enc_linear + enc_attn)

    # Decoder layer: self qkv/out, cross q + kv + out, MLP, causal self-attn, full cross-attn.
    dec_sa_linear = 2*T*Edec*(3*Edec) + 2*T*Edec*Edec
    dec_ca_linear = 2*T*Edec*(ca_nh*ca_h) + 2*S*Eenc*(2*ca_nh*ca_h) + 2*T*(ca_nh*ca_h)*Edec
    dec_mlp = 2*T*Edec*Hdec + 2*T*Hdec*Edec
    dec_sa_attn = 4*nh*h_dec*tgt_causal_pairs
    dec_ca_attn = 4*ca_nh*ca_h*cross_pairs
    fwd += dec_layers * (dec_sa_linear + dec_ca_linear + dec_mlp + dec_sa_attn + dec_ca_attn)

    # Decoder vocab projection. The current training loop computes logits for all tgt tokens,
    # then slices logits[:-1], so count all projected positions.
    fwd += 2*T*Edec*V
    train = 3*fwd
    return train, dict(S=S, T=T, src_lens=src_lens.tolist(), tgt_lens=tgt_lens.tolist(), src_sq=src_sq, tgt_causal_pairs=tgt_causal_pairs, cross_pairs=cross_pairs)

def do_microstep(batch, step):
    if random.random() < config.unconditional_frac:
        src_tokens = torch.zeros_like(batch['src_tokens'])
    else:
        src_tokens = batch['src_tokens']
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(#logits = model(
            src_ids=src_tokens,
            src_pos=batch['src_positions'],
            src_cu_seqlens=batch['src_cu_seqlens'],
            src_max_seqlen=batch['src_max_seqlen'],
            tgt_ids=batch['tgt_tokens'],
            tgt_pos=batch['tgt_positions'],
            tgt_cu_seqlens=batch['tgt_cu_seqlens'],
            tgt_max_seqlen=batch['tgt_max_seqlen'],
        )
        # loss = F.cross_entropy(logits[:-1, :], batch['tgt_tokens'][1:])
    (loss / config.grad_accum_steps).backward()
    if (step + 1) % config.grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    return loss.detach()

# Compile + warm up. First compiled step can take a while.
print('warmup_start')
for i in range(6):
    do_microstep(batches[i % len(batches)], i)
torch.cuda.synchronize()
optimizer.zero_grad(set_to_none=True)
print('warmup_done')

# Measured window.
measure_steps = 16
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
flops = 0
stats = []
start.record()
for i in range(measure_steps):
    batch = batches[i % len(batches)]
    f, st = flops_for_batch(batch)
    flops += f
    stats.append(st)
    do_microstep(batch, i)
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)

peak_bf16_tflops = 71.0  # RTX 3090 dense tensor-core FP16/BF16 class peak, not sparse.
achieved_tflops = flops / (elapsed_ms/1000) / 1e12
mfu = achieved_tflops / peak_bf16_tflops

print('RESULT')
print(f'measure_steps={measure_steps}')
print(f'elapsed_ms={elapsed_ms:.3f}')
print(f'microsteps_per_s={measure_steps/(elapsed_ms/1000):.4f}')
print(f'effective_steps_per_s={measure_steps/(elapsed_ms/1000)/config.grad_accum_steps:.4f}')
print(f'analytic_train_flops={flops}')
print(f'achieved_model_tflops={achieved_tflops:.3f}')
print(f'peak_bf16_tflops={peak_bf16_tflops:.1f}')
print(f'mfu={mfu:.5f}')
print('BATCH_STATS')
for i, st in enumerate(stats[:8]):
    print(f'batch={i} S={st["S"]} T={st["T"]} src_lens={st["src_lens"]} tgt_lens={st["tgt_lens"]} flops={flops_for_batch(batches[i % len(batches)])[0]}')

# Profiler cross-check: one measured step, CPU off to keep it lightweight. Torch profiler FLOPs mostly covers matmuls, not varlen attention kernels.
optimizer.zero_grad(set_to_none=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], with_flops=True, record_shapes=False) as prof:
    do_microstep(batches[0], 0)
torch.cuda.synchronize()
prof_flops = sum((getattr(e, 'flops', 0) or 0) for e in prof.key_averages())
ana0, st0 = flops_for_batch(batches[0])
print('PROFILER_CHECK')
print(f'profiler_reported_flops_batch0={prof_flops}')
print(f'analytic_train_flops_batch0={ana0}')
print(f'ratio_profiler_to_analytic={prof_flops/ana0 if ana0 else float("nan"):.5f}')
