# RWKV-7: https://github.com/BlinkDL/RWKV-LM
import math
import torch
from torch import nn
import torch.nn.functional as F
from fla.layers import RWKV7Attention # type: ignore

class TMix(nn.Module):
    def __init__(self, dim: int, block_id: int, n_blocks: int):
        super().__init__()
        # 根据BlinkDL原版实现中的建议取值
        decay_low_rank_dim = max(32, int(round(1.8 * (dim ** 0.5) / 32) * 32))
        a_low_rank_dim = max(32, int(round(1.8 * (dim ** 0.5) / 32) * 32))
        v_low_rank_dim = max(32, int(round(1.3 * (dim ** 0.5) / 32) * 32))
        gate_low_rank_dim = max(32, int(round(0.6 * (dim ** 0.8) / 32) * 32))
        self.rwkv7 = RWKV7Attention(
            "chunk",
            dim,
            layer_idx=block_id,
            decay_low_rank_dim=decay_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim
        )
        with torch.no_grad(): # 参数初始化，从BlinkDL原版实现中复制过来的
            def ortho_init(x, scale):
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x[i], gain=gain * scale)
                else:
                    assert False
                return x

            ratio_1_to_0 = 1 - (block_id / n_blocks)
            ratio_0_to_1 = block_id / (n_blocks - 1)
            ddd = torch.ones(dim)
            for i in range(dim):
                ddd[i] = i / dim

            x_r = 1.0 -  torch.pow(ddd, 0.2 * ratio_1_to_0)
            x_w = 1.0 -  torch.pow(ddd, 0.9 * ratio_1_to_0)
            x_k = 1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_0) + 0.4 * ratio_0_to_1)
            x_v = 1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_0) + 0.6 * ratio_0_to_1)
            x_a = 1.0 -  torch.pow(ddd, 0.9 * ratio_1_to_0)
            x_g = 1.0 -  torch.pow(ddd, 0.2 * ratio_1_to_0)
            x_x = torch.stack([x_r, x_w, x_k, x_v, x_a, x_g])
            self.rwkv7.x_x.data.copy_(x_x)

            w1 = torch.zeros(dim, decay_low_rank_dim)
            w2 = ortho_init(torch.zeros(decay_low_rank_dim, dim), 0.1)
            decay_speed = torch.ones(dim)
            for n in range(dim):
                decay_speed[n] = -7 + 5 * (n / (dim - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5) # WTF?
            w0 = decay_speed + 0.5
            self.rwkv7.w_lora.lora[0].weight.data.copy_(w1.T)
            self.rwkv7.w_lora.lora[2].weight.data.copy_(w2.T)
            self.rwkv7.w_lora.lora[2].bias.data.copy_(w0)

            a1 = torch.zeros(dim, a_low_rank_dim)
            a2 = ortho_init(torch.zeros(a_low_rank_dim, dim), 0.1)
            a0 = torch.zeros(dim)
            self.rwkv7.a_lora.lora[0].weight.data.copy_(a1.T)
            self.rwkv7.a_lora.lora[2].weight.data.copy_(a2.T)
            self.rwkv7.a_lora.lora[2].bias.data.copy_(a0)

            if block_id != 0: # 第一层没有这个模块
                v1 = torch.zeros(dim, v_low_rank_dim)
                v2 = ortho_init(torch.zeros(v_low_rank_dim, dim), 0.1)
                v0 = torch.zeros(dim) + 1.0
                self.rwkv7.v_lora.lora[0].weight.data.copy_(v1.T)
                self.rwkv7.v_lora.lora[2].weight.data.copy_(v2.T)
                self.rwkv7.v_lora.lora[2].bias.data.copy_(v0)
                del v1, v2, v0

            g1 = torch.zeros(dim, gate_low_rank_dim)
            g2 = ortho_init(torch.zeros(gate_low_rank_dim, dim), 0.1)
            self.rwkv7.g_lora.lora[0].weight.data.copy_(g1.T)
            self.rwkv7.g_lora.lora[2].weight.data.copy_(g2.T)

            self.rwkv7.r_proj.weight.data.uniform_(-0.5/(dim**0.5), 0.5/(dim**0.5))
            self.rwkv7.k_proj.weight.data.uniform_(-0.05/(dim**0.5), 0.05/(dim**0.5))
            self.rwkv7.v_proj.weight.data.uniform_(-0.5/(dim**0.5), 0.5/(dim**0.5))
            self.rwkv7.o_proj.weight.data.zero_()

            del ddd, x_r, x_w, x_k, x_v, x_a, x_g, x_x, w1, w2, w0, a1, a2, a0, g1, g2
    
    def forward(self, x: torch.Tensor, v_first: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x_attn, _, past_key_values, v_first = self.rwkv7(x, v_first=v_first)
        assert v_first is not None, "v_first should not be None"
        return x_attn, v_first

class CMix(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, block_id: int, n_blocks: int):
        super().__init__()
        self.dim = dim
        self.shift1 = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_0 = 1 - (block_id / n_blocks)
            # ddd = torch.linspace(0, 1, dim) * (dim - 1) / dim
            ddd = torch.ones(dim)
            for i in range(dim):
                ddd[i] = i / dim
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_0 ** 4))
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)

        self.key.weight.data.uniform_(-0.5/(dim**0.5), 0.5/(dim**0.5))
        self.value.weight.data.zero_()

        del ddd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.shift1(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

class Block(nn.Module):
    """一个Decoder块"""
    def __init__(self, dim: int, block_id: int, n_blocks: int):
        super().__init__()
        self.attn = TMix(dim, block_id, n_blocks)
        self.mlp = CMix(dim, dim * 4, block_id, n_blocks)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x_attn, v_first = self.attn(self.norm1(x), v_first=v_first)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        assert v_first is not None, "v_first should not be None"
        return x, v_first

class LLM(nn.Module):
    """大模型本体"""
    def __init__(self, vocab_size: int, dim: int,
                n_blocks: int, max_lr: float):
        assert dim % 64 == 0, "dim必须是64的倍数"
        assert math.log2(vocab_size).is_integer(), "vocab_size必须是2的幂"
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, i, n_blocks)
            for i in range(n_blocks)
        ])
        self.lmhead = nn.Linear(dim, vocab_size)
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.wte.weight.data.uniform_(-max_lr, max_lr)
        nn.init.orthogonal_(self.lmhead.weight, gain=0.5)

    def forward(self, x: torch.Tensor):
        x = self.norm_in(self.wte(x))
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first)
        return self.lmhead(self.norm_out(x))

    def save(self, path: str):
        torch.save(self.state_dict(), path) # 保存模型参数防止带上不必要的前缀
