from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out_proj(out)
        return self.dropout(out)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(config.d_model, config.mlp_hidden),
                nn.GELU(),
                nn.Linear(config.mlp_hidden, config.d_model),
                nn.Dropout(config.dropout),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
            self,
            idx: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            return_hidden_states: bool = False,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, List[torch.Tensor]]]]:
        _b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.config.block_size}")

        pos = torch.arange(0, t, device=idx.device, dtype=torch.long)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos).unsqueeze(0)
        x = tok_emb + pos_emb

        hidden_states: Optional[Dict[str, List[torch.Tensor]]] = None
        layer_states: Optional[List[torch.Tensor]] = None
        if return_hidden_states:
            layer_states = []

        for block in self.blocks:
            x = block(x)
            if layer_states is not None:
                layer_states.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        if layer_states is not None:
            hidden_states = {"layers": layer_states}

        return logits, loss, hidden_states

    @torch.no_grad()
    def generate(
            self,
            idx: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _loss, _hidden = self(idx_cond)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))
                top_vals, _top_idx = torch.topk(next_token_logits, k=k, dim=-1)
                threshold = top_vals[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold, float("-inf"))

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
