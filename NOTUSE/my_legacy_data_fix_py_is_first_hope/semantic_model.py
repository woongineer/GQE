# ===========================
# model.py
# ===========================
"""
small model!  (hierarchical head version)
"""
import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def _ensure_pos_emb(self, T_needed):
        cur = self.pos_emb.num_embeddings
        if T_needed > cur:
            extra = T_needed - cur
            new_emb = nn.Embedding(T_needed, self.config.n_embd).to(self.pos_emb.weight.device)
            # init new
            torch.nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
            # copy old
            new_emb.weight.data[:cur] = self.pos_emb.weight.data
            self.pos_emb = new_emb

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 32
    vocab_size: int = 64  # (미사용: 기존 단일 토큰 방식 잔재)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True

    ###########수정된 부분##########
    num_qubits: int = 4
    num_gates: int = 6            # ['RX','RY','RZ','CNOT','H','I']
    num_params: int = 4           # feature index 수
    pad_idx: int = -1             # p, t 미사용 시
    ###########수정된 부분##########


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        # 기존 토큰 임베딩은 사용 안 함. 대신 component-wise embedding 사용
        ###########수정된 부분##########
        self.embed_q = nn.Embedding(config.num_qubits + 1, config.n_embd)   # +1 for BOS/PAD(=0)
        self.embed_g = nn.Embedding(config.num_gates + 1, config.n_embd)
        self.embed_p = nn.Embedding(config.num_params + 1, config.n_embd)
        self.embed_t = nn.Embedding(config.num_qubits + 1, config.n_embd)
        ###########수정된 부분##########

        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        ###########수정된 부분##########
        # Factorized heads
        self.head_q = nn.Linear(config.n_embd, config.num_qubits, bias=config.bias)
        self.head_g = nn.Linear(config.n_embd, config.num_gates, bias=config.bias)
        self.head_p = nn.Linear(config.n_embd, config.num_params, bias=config.bias)
        self.head_t = nn.Linear(config.n_embd, config.num_qubits, bias=config.bias)
        ###########수정된 부분##########

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # remove positional embedding? keep as is (minor)
            pass
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.pos_emb.weight = nn.Parameter(self.pos_emb.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                      **(dict(fused=True) if use_fused else {}))
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    ###########수정된 부분##########
    def forward(self, q_ids, g_ids, p_ids, t_ids):
        """
        q_ids,g_ids,p_ids,t_ids: (B,T) with 0 = BOS/PAD, >0 are valid indices.
        """
        B, T = q_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=q_ids.device)

        # shift index (+1) already handled outside. Here assume 0 is PAD/BOS
        q_emb = self.embed_q(q_ids)
        g_emb = self.embed_g(g_ids)
        p_emb = self.embed_p(p_ids)
        t_emb = self.embed_t(t_ids)

        x = q_emb + g_emb + p_emb + t_emb + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        h = self.ln_f(x)

        q_logits = self.head_q(h)
        g_logits = self.head_g(h)
        p_logits = self.head_p(h)
        t_logits = self.head_t(h)
        return {"h": h, "q": q_logits, "g": g_logits, "p": p_logits, "t": t_logits}
    ###########수정된 부분##########
