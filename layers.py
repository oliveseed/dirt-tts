'''
definition of the tts backbone in pytorch, with both training and inference compatibility
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
import math


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.compute_dtype = dtype

        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / embedding_dims
        timescale = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)
        self.register_buffer("timescale", timescale, persistent=False)

    def forward(self, inputs: torch.Tensor, position: torch.Tensor):
        """Applies RoPE."""
        position = position.unsqueeze(-1).unsqueeze(-1)
        sinusoid_inp = position / self.timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        first_half, second_half = torch.chunk(inputs.to(torch.float32), 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat(
            (first_part.to(self.compute_dtype), second_part.to(self.compute_dtype)),
            dim=-1,
        )

    def apply_rope(self, inputs: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        first_half, second_half = torch.chunk(inputs.to(torch.float32), 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat((first_part.to(self.compute_dtype), second_part.to(self.compute_dtype)), dim=-1)


class KVCache:
    '''
    Key-Value Cache for storing past key and value tensors during autoregressive decoding.
    Not used during training. batch size is hard coded to 2 for CFG at inference.
    '''
    def __init__(self, num_heads, max_len, head_dim, device, k=None, v=None):
        self.k = torch.zeros((2, num_heads, max_len, head_dim), device=device) if k is None else k
        self.v = torch.zeros((2, num_heads, max_len, head_dim), device=device) if v is None else v
        self.current_idx = 0
        self.max_len = max_len

    def get_kv_for_attention(self, current_k, current_v):
        if self.current_idx == 0:
            return current_k, current_v
        else:
            past_k = self.k[:, :, : self.current_idx, :]
            past_v = self.v[:, :, : self.current_idx, :]
            attn_k = torch.cat((past_k, current_k), dim=2)
            attn_v = torch.cat((past_v, current_v), dim=2)
            return attn_k, attn_v

    def update_cache(self, k, v):
        assert self.current_idx < self.max_len
        self.k[:, :, self.current_idx : self.current_idx + 1, :] = k
        self.v[:, :, self.current_idx : self.current_idx + 1, :] = v
        self.current_idx += 1

    def prefill(self, k, v):
        prefill_len = k.shape[2]
        assert prefill_len <= self.max_len
        self.k[:, :, :prefill_len, :] = k
        self.v[:, :, :prefill_len, :] = v
        self.current_idx = prefill_len


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, 2*hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, in_dim, bias=False)
    def forward(self, x):
        x = self.in_proj(x)
        gate, up = x.chunk(2, dim=-1)
        return self.out_proj(up * F.silu(gate))


class GQA(nn.Module):
    def __init__(
        self,
        q_embed_dim,
        kv_embed_dim,
        out_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        compute_dtype,
    ):
        super().__init__()
        self.q_embed_dim = q_embed_dim
        self.kv_embed_dim = kv_embed_dim
        self.out_embed_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_gqa_groups = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(q_embed_dim, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(kv_embed_dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, out_embed_dim, bias=False)

        self.rotary_emb = RotaryEmbedding(
            embedding_dims=head_dim,
            min_timescale=1.0,
            max_timescale=10000.0,
            dtype=torch.float32,
        )

    def forward(
        self,
        x,
        q_pos,
        kv_pos=None,
        attn_mask=None,
        cache=None,
        prefill=False,
    ):
        if kv_pos is None:
            kv_pos = q_pos

        batch_size, seq_len, _ = x.shape
        orig_dtype = x.dtype

        q = self.q_proj(x).view(batch_size, seq_len, self.num_query_heads, self.head_dim)   # (B, T, N, H)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.rotary_emb(q, q_pos)
        k = self.rotary_emb(k, kv_pos)

        attn_k: torch.Tensor | None = cache.k if cache is not None else None
        attn_v: torch.Tensor | None = cache.v if cache is not None else None
        new_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is None:
            attn_k, attn_v = k, v
        elif prefill:
            attn_k, attn_v = k, v
            cache.prefill(attn_k, attn_v)
        else:
            new_kv_cache = k, v
            attn_k, attn_v = cache.get_kv_for_attention(k, v)

        attn_output = F.scaled_dot_product_attention(
            q,
            attn_k,
            attn_v,
            attn_mask=attn_mask,
            scale=1.0,
            enable_gqa=self.num_gqa_groups > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_embed_dim)
        output = self.out_proj(attn_output)

        return output.to(orig_dtype), new_kv_cache


class GQCA(nn.Module):
    def __init__(
        self,
        q_embed_dim,
        kv_embed_dim,
        out_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        compute_dtype,
    ):
        super().__init__()
        self.q_embed_dim = q_embed_dim
        self.kv_embed_dim = kv_embed_dim
        self.out_embed_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_gqa_groups = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(q_embed_dim, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(kv_embed_dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, out_embed_dim, bias=False)

        self.rotary_emb = RotaryEmbedding(
            embedding_dims=head_dim,
            min_timescale=1.0,
            max_timescale=10000.0,
            dtype=torch.float32,
        )

    def forward(
        self,
        xq,
        xkv,
        q_pos,
        kv_pos=None,
        attn_mask=None,
        cache=None,         #
    ):
        batch_size, seq_len_q, _ = xq.shape
        orig_dtype = xq.dtype

        if cache is not None:
            q = self.q_proj(xq).view(batch_size, seq_len_q, self.num_query_heads, self.head_dim)   # (B, T, N, H)
            q = self.rotary_emb(q, q_pos)
            
            attn_k: torch.Tensor | None = cache.k if cache is not None else None
            attn_v: torch.Tensor | None = cache.v if cache is not None else None
            q = q.transpose(1, 2)
            
            attn_output = F.scaled_dot_product_attention(
                q,
                attn_k,
                attn_v,
                attn_mask=attn_mask,
                scale=1.0,
                enable_gqa=self.num_gqa_groups > 1,
            )

        else:
            _, seq_len_kv, _ = xkv.shape

            q = self.q_proj(xq).view(batch_size, seq_len_q, self.num_query_heads, self.head_dim)   # (B, T, N, H)
            k = self.k_proj(xkv).view(batch_size, seq_len_kv, self.num_kv_heads, self.head_dim)
            v = self.v_proj(xkv).view(batch_size, seq_len_kv, self.num_kv_heads, self.head_dim)

            q = self.rotary_emb(q, q_pos)
            k = self.rotary_emb(k, kv_pos)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                scale=1.0,
                enable_gqa=self.num_gqa_groups > 1,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.q_embed_dim)   # note
        output = self.out_proj(attn_output)

        return output.to(orig_dtype)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        expand_dim,
        norm_eps,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.pre_norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )
        self.gqa = GQA(
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            out_embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            compute_dtype=torch.float32,
        )
        self.post_norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )
        self.mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=expand_dim,
        )

    def forward(
        self,
        x,
        q_pos,
        kv_pos=None,
        attn_mask=None,
        cache=None,
        prefill=False,
    ):
        residual = x
        x_norm = self.pre_norm(x)
        attn_out, _ = self.gqa(
            x_norm,
            q_pos,
            kv_pos,
            attn_mask,
            cache,
            prefill,
        )
        x = residual + attn_out
        residual = x
        x_norm = self.post_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        expand_dim,
        norm_eps,
        vocab_size,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                expand_dim=expand_dim,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )

    def forward(
        self,
        x_ids,
        q_pos,
        kv_pos=None,
        attn_mask=None,
        cache=None,
        prefill=False,
    ):
        x = self.embedding(x_ids)
        for layer in self.layers:
            x = layer(
                x,
                q_pos,
                kv_pos,
                attn_mask,
                cache,
                prefill,
            )
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        enc_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        ca_num_query_heads,
        ca_num_kv_heads,
        ca_head_dim,
        expand_dim,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps

        self.pre_sa_norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )
        self.pre_ca_norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )
        self.pre_mlp_norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )

        self.sa = GQA(
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            out_embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            compute_dtype=torch.float32,
        )
        self.ca = GQCA(
            q_embed_dim=embed_dim,
            kv_embed_dim=enc_embed_dim,
            out_embed_dim=embed_dim,
            num_query_heads=ca_num_query_heads,
            num_kv_heads=ca_num_kv_heads,
            head_dim=ca_head_dim,
            compute_dtype=torch.float32,
        )
        self.mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=expand_dim,
        )

    def forward(
        self,
        x,
        enc_out,
        dec_q_pos,
        dec_kv_pos,
        enc_kv_pos,
        sa_cache=None,
        ca_cache=None,
        sa_mask=None,
        ca_mask=None,
        prefill=False,
    ):
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out, new_kv_cache = self.sa(
            x=x_norm,
            q_pos=dec_q_pos,
            kv_pos=dec_kv_pos,
            attn_mask=sa_mask,
            cache=sa_cache,
            prefill=prefill,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.ca(
            xq=x_norm,
            xkv=enc_out,
            q_pos=dec_q_pos,
            kv_pos=enc_kv_pos,
            attn_mask=ca_mask,
            cache=ca_cache,
        )
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x, new_kv_cache


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        enc_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        ca_num_query_heads,
        ca_num_kv_heads,
        ca_head_dim,
        expand_dim,
        norm_eps,
        vocab_size,
        num_channels,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_channels = num_channels
        assert embed_dim // num_query_heads == head_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(num_channels)
        ])
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                enc_embed_dim=enc_embed_dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                ca_num_query_heads=ca_num_query_heads,
                ca_num_kv_heads=ca_num_kv_heads,
                ca_head_dim=ca_head_dim,
                expand_dim=expand_dim,
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )

        self.logits_proj = nn.Linear(embed_dim, num_channels * vocab_size, bias=False)

    def precompute_cross_attn_cache(
        self,
        max_len: int,
        enc_out: torch.Tensor,  # (B, S, E)
        src_positions: torch.Tensor,
    ) -> list[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
        only used at inference.
        """
        per_layer_kv_cache: list[KVCache] = []

        batch_size, seq_len, _ = enc_out.shape

        for layer in self.layers:
            cross_attn_module = layer.ca
            k_proj = cross_attn_module.k_proj(enc_out).view(batch_size, seq_len, cross_attn_module.num_kv_heads, cross_attn_module.head_dim)
            v_proj = cross_attn_module.v_proj(enc_out).view(batch_size, seq_len, cross_attn_module.num_kv_heads, cross_attn_module.head_dim)

            k_proj = cross_attn_module.rotary_emb(k_proj, position=src_positions)
            k = k_proj.transpose(1, 2)
            v = v_proj.transpose(1, 2)

            per_layer_kv_cache.append(
                KVCache(
                    num_heads=cross_attn_module.num_kv_heads,
                    max_len=max_len,
                    head_dim=cross_attn_module.head_dim,
                    device=k.device,
                    k=k,
                    v=v,
                )
            )

        return per_layer_kv_cache

    def decode_step(
        self,
        tgt_ids,
        enc_out,
        tgt_pos,
        src_pos,
        sa_mask,
        ca_mask,
        sa_cache,
        ca_cache,
    ):
        '''
        only used at inference.
        caches should not be None
        '''

        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        new_cache = []

        for i, layer in enumerate(self.layers):
            self_cache = sa_cache[i]
            cross_cache = ca_cache[i]
            x, new_kv_cache = layer(
                x=x,
                enc_out=None, # or enc_out if not using kvcache
                dec_q_pos=tgt_pos,
                dec_kv_pos=tgt_pos,
                enc_kv_pos=src_pos,
                sa_cache=self_cache,
                ca_cache=cross_cache,
                sa_mask=sa_mask,
                ca_mask=ca_mask,
            )
            new_cache.append(new_kv_cache)
        x = self.norm(x)
        logits = self.logits_proj(x)
        return logits, new_cache
  
    def forward(
        self,
        tgt_ids,
        enc_out,
        tgt_pos,
        src_pos,
        sa_mask,
        ca_mask,
        sa_cache,
        ca_cache,
    ):
        '''
        only used at training.
        caches are set to None and it should still work
        '''

        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            x, _ = layer(
                x=x,
                enc_out=enc_out,
                dec_q_pos=tgt_pos,
                dec_kv_pos=tgt_pos,
                enc_kv_pos=src_pos,
                sa_cache=None,
                ca_cache=None,
                sa_mask=sa_mask,
                ca_mask=ca_mask,
                #prefill=True,   #
            )
        x = self.norm(x)
        logits = self.logits_proj(x)
        return logits


class DirtModel(nn.Module):
    def __init__(
        self,
        enc_num_layers,
        enc_embed_dim,
        enc_hidden_dim,
        enc_head_dim,
        dec_num_layers,
        dec_embed_dim,
        dec_hidden_dim,
        dec_head_dim,
        num_query_heads,
        num_kv_heads,
        ca_num_query_heads,
        ca_num_kv_heads,
        ca_head_dim,
        norm_eps=1e-5,
        text_vocab_size=256,
        audio_vocab_size=46660,
        num_channels=1,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers=enc_num_layers,
            embed_dim=enc_embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=enc_head_dim,
            expand_dim=enc_hidden_dim,
            norm_eps=norm_eps,
            vocab_size=text_vocab_size,
        )
        self.decoder = Decoder(
            num_layers=dec_num_layers,
            embed_dim=dec_embed_dim,
            enc_embed_dim=enc_embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=dec_head_dim,
            ca_num_query_heads=ca_num_query_heads,
            ca_num_kv_heads=ca_num_kv_heads,
            ca_head_dim=ca_head_dim,
            expand_dim=dec_hidden_dim,
            norm_eps=norm_eps,
            vocab_size=audio_vocab_size,
            num_channels=num_channels,
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.modules.normalization.RMSNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        src_ids,
        src_pos,
        tgt_ids,
        tgt_pos,
        enc_sa_mask,
        dec_sa_mask,
        dec_ca_mask,
    ):
        '''
        only used at training.
        caches are set to None and it should still work
        '''

        enc_out = self.encoder(
            src_ids,
            src_pos,
            attn_mask=enc_sa_mask,
        )

        B, T, C = tgt_ids.shape
        device = tgt_ids.device

        sa_cache = None
        ca_cache = None

        dec_out = self.decoder(
            tgt_ids,
            enc_out,
            tgt_pos,
            src_pos,
            sa_mask=dec_sa_mask,
            ca_mask=dec_ca_mask,
            sa_cache=sa_cache,
            ca_cache=ca_cache,
        )
        return dec_out
