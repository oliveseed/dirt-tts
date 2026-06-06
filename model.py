'''
wrapper of the model for inference
'''

import torch
import torchaudio
import numpy as np
import time
from collections import Counter
from transformers import AutoTokenizer

from stable_codec import StableCodec
from layers import TTSModel, KVCache


def clean_prompt(text_prompt):
    # replace typographic quotes with straight quotes
    translation_table = str.maketrans({
        '‘': "'",
        '’': "'",
        '“': '"',
        '”': '"',
    })
    normalized_prompt = text_prompt.translate(translation_table)
    # TODO: possibly do more normalizations
    return normalized_prompt

def apply_repetition_penalty(logits, generated, penalty, per_tok_freq):
    if penalty == 1.0: return

    batch_size, vocab_size = logits.shape
    device = logits.device

    multipliers = torch.ones((batch_size, vocab_size), dtype=logits.dtype, device=device)
    for b, seq in enumerate(generated):
        if not seq: continue
        if per_tok_freq:
            counts = Counter(seq)
            for token_id, freq in counts.items():
                if 0 <= token_id < vocab_size:
                    multipliers[b, token_id] = penalty ** freq
        else:
            for token_id in set(seq):
                if 0 <= token_id < vocab_size:
                    multipliers[b, token_id] = penalty

    neg_mask  = (logits < 0).to(logits.dtype)
    pos_mask = 1.0 - neg_mask

    logits.mul_(1.0 + (multipliers - 1.0) * neg_mask)
    logits.div_(1.0 + (multipliers - 1.0) * pos_mask)


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


class TTS:
    def __init__(self, tts_config: dict):
        model_cfg = tts_config['model']['architecture']
        codec_cfg = tts_config['audio_codec']

        self.model = TTSModel(**model_cfg)
        self.text_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

        self.codec = StableCodec(
            model_config_path=codec_cfg['config_path'],
            ckpt_path=codec_cfg['ckpt_path'],
            device=torch.device('cuda'), # put it on the gpu to begin with or it won't work
        )
        self.codec.set_posthoc_bottleneck(codec_cfg['posthoc_bottleneck'])

        self.max_text_len = 123 #
        self.text_eos, self.text_pad = 2, 0
        
        self.audio_eos = 46657
        self.audio_pad = 0
        self.audio_bos = 46658

    def load_weights(self, ckpt_path):
        pass
        # state_dict = torch.load(ckpt_path, map_location="cpu")
        # # remove _orig_mod prefix because torch.compile was used during training
        # unwanted_prefix = '_orig_mod.'
        # for k,v in list(state_dict.items()):
        #     if k.startswith(unwanted_prefix):
        #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Decode one sequence of audio tokens into waveform

        Input:
            codes               (len, 1)
        Returns:
            decoded_audio_pt    (1, 1, waveform_len) tensor
            decoded_audio_np    (waveform_len) np array
        """

        codes = [codes.transpose(1, 0).unsqueeze(-1).to(self.codec.device)]
        decoded_audio_pt = self.codec.decode(codes, posthoc_bottleneck=True)

        decoded_audio_np = decoded_audio_pt.squeeze(0).squeeze(0).cpu().numpy()
        decoded_audio_np = (decoded_audio_np / np.abs(decoded_audio_np).max() * 32767).astype(np.int16)

        return decoded_audio_pt, decoded_audio_np

    @torch.no_grad()
    def _generate(
        self,
        prompt,
        batch_size: int = 1,
        max_audio_len: int = 768,
        cfg_scale: float = 3.0,
        temperature: float = 1.1,
        top_p: float = 0.95,
        use_cfg_filter: bool = True,
        cfg_filter_top_k: int = 35, 
        repetition_penalty: float = 1.0,
        device: torch.device = 'cuda',
    ):
        """
        Generate a batch of audio token sequences given one text prompt. The decoding loop runs
        until every sequence is done generating. Because of CFG, the actual batch size will be doubled.
        There is duplicated kv cache which can be fixed in the future

        1) encoder pass
        2) decoder cache setup
        3) decoder output initialization
        4) decoding loop

        Returns:
            out_codes       (batch_size, batch_max_len, 1)
            out_lengths     (batch_size)
        """

        # t0 = time.time()

        # max_text_len = 123 #
        actual_bs = 2 * batch_size

        # text_eos, text_pad = 2, 0
        audio_eos, audio_pad, audio_bos = self.audio_eos, self.audio_pad, self.audio_bos

        # 1) encoder pass
        if type(prompt) == torch.Tensor:
            # text_tokens = prompt[:self.max_text_len-1]
            # src_tok = text_tokens.expand(batch_size, -1)
            src_tok = prompt
        # elif type(prompt) == str:
        #     text_tokens = self.text_tokenizer(prompt, add_special_tokens=False)['input_ids'][:max_text_len-1] + [text_eos]
        #     src_tok = torch.tensor(text_tokens, device=device).expand(batch_size, -1)
        else:
            print(f"{type(prompt)} unsupported")

        L = src_tok.size(-1)
        enc_input_uncond = torch.zeros(batch_size, L, dtype=torch.long, device=device)
        enc_input_cond = src_tok
        enc_input = torch.cat([enc_input_uncond, enc_input_cond], dim=0) # (2B, L)

        enc_pos = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(0).expand(actual_bs, L)
        padding_mask = (enc_input_cond.squeeze(1) != self.text_pad).to(device).repeat_interleave(2, dim=0)
        enc_attn_mask = (padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)).unsqueeze(1)

        enc_out = self.model.encoder.forward_inference(
            x_ids=enc_input,
            pos=enc_pos,
            attn_mask=enc_attn_mask,
        )

        # 2) decoder pass: caches setup
        dec_ca_cache = self.model.decoder.precompute_cross_attn_cache(
            max_len=max_audio_len,
            enc_out=enc_out, 
            src_positions=enc_pos,
        )
        dec_sa_cache = [
            KVCache(
                batch_size=actual_bs,
                num_heads=self.model.decoder.layers[0].sa.num_kv_heads,
                max_len=max_audio_len,
                head_dim=self.model.decoder.layers[0].sa.head_dim, # 64
                device=device,
            ) for _ in range(len(self.model.decoder.layers))
        ]

        # 3) decoder pass: initialize generation
        generated = torch.full(
            (actual_bs, 1, 1),
            fill_value=audio_bos,
            dtype=torch.long,
            device=device,
        )
        generated = torch.cat([
            generated,
            torch.full(
                (actual_bs, max_audio_len, 1),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            ),
        ], dim=1)
        
        tgt_padding_mask = (
            (generated[:, -1, :].unsqueeze(1) != audio_pad).any(dim=2).to(device)
        )
        dec_ca_mask = (tgt_padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)).unsqueeze(1)

        # 4) decoder pass: run decoding loop
        reached_end = torch.zeros(actual_bs, dtype=torch.bool, device=device)
        generated_lens = torch.full((actual_bs,), fill_value=max_audio_len-1, device=device)
        generated_arr = []

        for step in range(max_audio_len-1):
            tgt_ids = generated[:, step, :]
            tgt_pos = torch.full(
                (actual_bs, 1),
                fill_value=step,
                dtype=torch.long,
                device=device,
            )

            logits, new_cache = self.model.decoder.decode_step(
                tgt_ids=tgt_ids,
                enc_out=enc_out,
                tgt_pos=tgt_pos,
                src_pos=enc_pos,
                sa_mask=None,
                ca_mask=dec_ca_mask,
                sa_cache=dec_sa_cache,
                ca_cache=dec_ca_cache,
            )
            V = logits.size(-1)
            logits = logits.reshape(actual_bs, 1, V)
            
            logits_last = logits[:, -1, :]          # (2B, V)
            uncond_logits, cond_logits = logits_last.chunk(2, dim=0)

            cfg_logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)
            cfg_logits[:, audio_pad] = -torch.inf

            # apply_repetition_penalty(logits, [generated_arr], penalty=repetition_penalty, per_tok_freq=True)

            pred = _sample_next_token(
                cfg_logits.float(),
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=use_cfg_filter,
                cfg_filter_top_k=cfg_filter_top_k,
            )
            # generated_arr.append(pred.item())
            next_tokens = torch.cat([pred, pred], dim=0).unsqueeze(-1)
            generated[:, step+1, :] = next_tokens
            
            reached_end |= (next_tokens.squeeze(-1) == audio_eos)
            generated_lens[reached_end] = torch.clamp_max(
                generated_lens[reached_end], step
            )
            if torch.all(reached_end):
                # print(pred)
                break

            for i, layer_cache in enumerate(dec_sa_cache):
                layer_cache.update_cache(new_cache[i][0], new_cache[i][1])

        gen_codes = generated[:batch_size, :step+2, :]        
        gen_lengths = generated_lens[:batch_size]

        return gen_codes, gen_lengths

    def generate(
        self,
        prompt,
        batch_size: int = 1,
        max_audio_len: int = 768,
        cfg_scale: float = 3.0,
        temperature: float = 1.1,
        top_p: float = 0.95,
        use_cfg_filter: bool = True,
        cfg_filter_top_k: int = 35, 
        repetition_penalty: float = 1.0,
        device: torch.device = 'cuda',
    ):
        prompt = clean_prompt(prompt)
        
        text_tokens = self.text_tokenizer(prompt, add_special_tokens=False)['input_ids']
        text_tokens_eos = text_tokens[:self.max_text_len-1] + [self.text_eos]
        text_tokens_pt = torch.tensor(text_tokens_eos, device=device).expand(batch_size, -1)

        prompt_truncated = False
        if len(text_tokens_eos) <= len(text_tokens):
            prompt_truncated = True
            print("prompt exceeded max length and was truncated")

        with torch.autocast('cuda', dtype=torch.bfloat16):
            gen_codes, gen_lens = self._generate(
                prompt=text_tokens_pt,
                batch_size=batch_size,
                max_audio_len=max_audio_len,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=use_cfg_filter,
                cfg_filter_top_k=cfg_filter_top_k,
                repetition_penalty=repetition_penalty,
            )

        # for i, seq in enumerate(gen_codes):
        #     waveform, waveform_np = self.decode(seq[1:lens[i]], device=device)

        # waveform, waveform_np = self.decode(gen_codes[0, 1:gen_lens[0]], device=device)
        # print(prompt)
        # print(waveform_np.shape)
        # return waveform_np

        return gen_codes, gen_lens, prompt_truncated
