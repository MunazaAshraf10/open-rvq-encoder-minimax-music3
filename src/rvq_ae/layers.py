"""Pre norm transformer layer shared by the temporal encoder and the depth decoder."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float,
    causal: bool,
    dropout: float,
    training: bool,
) -> Tensor:
    """Scaled dot product attention with the softmax evaluated in float32.

    Inputs are [batch, heads, length, head_dim]; the scale is passed in explicitly because muP
    uses 1 / d_head rather than the 1 / sqrt(d_head) default.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        length = scores.shape[-1]
        mask = torch.ones(length, length, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    probs = F.dropout(probs, p=dropout, training=training)
    return torch.matmul(probs, v)


class Layer(nn.Module):
    """x + Attn(LN(x)); x + FFN(LN(x)) with GELU and separate q, k, v, out projections."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        *,
        scale: float,
        causal: bool,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.scale = scale
        self.causal = causal
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * ff_mult)
        self.linear2 = nn.Linear(dim * ff_mult, dim)

    def split(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        return x.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)

    def attend(self, x: Tensor) -> Tensor:
        q, k, v = self.split(self.q_proj(x)), self.split(self.k_proj(x)), self.split(self.v_proj(x))
        out = attention(
            q, k, v, scale=self.scale, causal=self.causal, dropout=self.dropout, training=self.training
        )
        return self.out_proj(out.transpose(1, 2).reshape(x.shape))

    def feed_forward(self, x: Tensor) -> Tensor:
        hidden = F.dropout(F.gelu(self.linear1(x)), p=self.dropout, training=self.training)
        return self.linear2(hidden)

    def forward(self, x: Tensor) -> Tensor:
        x = x + F.dropout(self.attend(self.norm1(x)), p=self.dropout, training=self.training)
        return x + F.dropout(self.feed_forward(self.norm2(x)), p=self.dropout, training=self.training)
