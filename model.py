'''
the full model including the audio tokenizer
'''

import torch
import torchaudio
import numpy as np
import time

from stable_codec import StableCodec
from layers import DirtModel, KVCache


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    use_cfg_filter: bool,
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if use_cfg_filter and cfg_filter_top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        # Calculate indices to remove based on top_p
        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        # Shift the mask to the right to keep the first token above the threshold
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[..., :-1].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0  # Always keep the most probable token

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C

def _basic_sample(logits, temperature, top_k):
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return idx_next


class DirtTTS:
    def __init__(self, dirt_config: dict):

        model_cfg = dirt_config['model']['architecture']
        codec_cfg = dirt_config['tokenizer']
        self.device = dirt_config['model']['device']
        self.codec_device = dirt_config['tokenizer']['device']

        self.vocab_size = model_cfg['audio_vocab_size']
        self.text_pad_val = dirt_config['model']['text_pad_id']
        self.eos_tok = dirt_config['model']['eos_tok_id']
        self.pad_tok = dirt_config['model']['pad_tok_id']
        self.bos_tok = dirt_config['model']['bos_tok_id']

        # load backbone
        state_dict = torch.load(dirt_config['model']['ckpt_path'], map_location="cpu")
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model = DirtModel(**model_cfg)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # load tokenizer
        self.codec = StableCodec(
            model_config_path=codec_cfg['config_path'],
            ckpt_path=codec_cfg['ckpt_path'],
            device=torch.device(self.codec_device)
        )
        self.codec.set_posthoc_bottleneck(codec_cfg['posthoc_bottleneck'])

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        codes = [tokens.transpose(1, 0).unsqueeze(-1).to(self.device)]
        decoded_audio = self.codec.decode(codes, posthoc_bottleneck=True)
        decoded_audio = decoded_audio.squeeze(0).squeeze(0).cpu().numpy()
        decoded_audio = (decoded_audio / np.abs(decoded_audio).max() * 32767).astype(np.int16)
        return decoded_audio

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_audio_len,
        cfg_scale,
        temperature,
        top_p,
        cfg_filter_top_k, 
    ):
        t0 = time.time()

        max_len = 512
        text_pad_val = self.text_pad_val
        eos_tok = self.eos_tok
        pad_tok = self.pad_tok
        bos_tok = self.bos_tok        
        bs = 1

        # encode text
        text = list(prompt.encode('utf-8'))
        text = torch.tensor(text[:max_len], dtype=torch.long, device=self.device)
        text = [text]
        src_tok = torch.full(
            (bs, 1, max_len),
            fill_value=text_pad_val,
            dtype=torch.long,
            device=self.device,
        )
        current_len = len(text[0])
        src_tok[0, 0, :current_len] = text[0]
        text = src_tok

        enc_input_uncond = torch.zeros_like(text)
        enc_input_cond = text
        stacked_inputs = torch.stack([enc_input_uncond, enc_input_cond], dim=1)
        enc_input = stacked_inputs.view(2 * bs, -1)

        positions = torch.arange(max_len, dtype=torch.float32, device=self.device).unsqueeze(0)
        padding_mask = (enc_input_cond.squeeze(1) != 0).to(self.device).repeat_interleave(2, dim=0)
        attn_mask = (padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)).unsqueeze(1)
        enc_out = self.model.encoder(
            x_ids=enc_input,
            q_pos=positions,
            attn_mask=attn_mask,
        )

        # decoder setup
        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(
            max_len=max_audio_len,
            enc_out=enc_out, 
            src_positions=positions,
        )
        self_attn_cache = [
            KVCache(
                num_heads=16,
                max_len=max_audio_len,
                head_dim=64,
                device=self.device,
            ) for _ in range(len(self.model.decoder.layers))
        ]
        generated = torch.full(
            (2, 1, 1),
            fill_value=bos_tok,
            dtype=torch.long,
            device=self.device,
        )
        curr_step = 0
        eos_detected = False
        eos_countdown = -1
        generated = torch.cat(
            [
                generated,
                torch.full(
                    (2, max_audio_len, 1),
                    fill_value=-1,
                    dtype=torch.long,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        tgt_padding_mask = (
            (generated[:, -1, :].unsqueeze(1) != pad_tok).any(dim=2).to(self.device)
        )
        dec_cross_attn_mask = (tgt_padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)).unsqueeze(1)

        # decoding loop
        for step in range(curr_step, curr_step + max_audio_len):
            tgt_ids = generated[:, step, :].unsqueeze(1)
            tgt_pos = torch.full(
                (2, 1),
                fill_value=step,
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                logits, new_cache = self.model.decoder.decode_step(
                    tgt_ids,
                    enc_out,
                    tgt_pos,
                    positions,
                    None,
                    dec_cross_attn_mask,
                    self_attn_cache,
                    dec_cross_attn_cache,
                    #current_idx,
                )
            logits = logits.reshape(2*bs, 1, 1, logits.size(-1))
            
            for i, layer_cache in enumerate(self_attn_cache):
                layer_cache.update_cache(new_cache[i][0], new_cache[i][1])
            
            V = self.vocab_size
            logits_last = logits[:, -1, :, :]
            uncond_logits = logits_last[0, :, :]
            cond_logits = logits_last[1, :, :]

            cfg_logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)

            logits = cfg_logits.reshape((-1, V))
            logits[:, pad_tok:] = -torch.inf

            pred = _sample_next_token(
                logits.float(),
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=True,
                cfg_filter_top_k=cfg_filter_top_k,
            )
            # pred = _basic_sample(
            #     logits.float(),
            #     temperature=temperature,
            #     top_k=cfg_filter_top_k,
            # )

            generation_step_index = step - curr_step
            generated[:, step+1, :] = pred.unsqueeze(0).expand(2, -1)
            if not eos_detected and pred[0] == eos_tok:
                eos_detected = True
                break
            generation_step_index = step - curr_step + 1

        out_codes = generated[:, 1:step+1, :]
        gen_codes = out_codes[0]
        decoded_audio = self.decode(gen_codes)
        t1 = time.time()
        print(f"time: {(t1-t0):.5}", f"RTF: {((t1-t0) / (decoded_audio.shape[0] / 16000)):.5}")
        print(decoded_audio.shape, decoded_audio)
        return (16000, decoded_audio)
