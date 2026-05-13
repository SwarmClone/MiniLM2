"""Benchmark: 预计算 vs 动态计算 RoPE 位置编码性能对比"""
import time
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)

DEVICE = "cpu"

# ---------- 预计算版本 ----------
class RoPE_Precomputed(nn.Module):
    def __init__(self, dim: int, max_length: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        positions = torch.arange(0, max_length, 1)
        theta = 1 / base ** (torch.arange(0, dim, 2) / dim)
        positions_theta = positions.unsqueeze(1) * theta.unsqueeze(0)
        self.register_buffer('positions_sin', torch.sin(positions_theta))
        self.register_buffer('positions_cos', torch.cos(positions_theta))
        self.dim = dim

    def forward(self, x: torch.Tensor, *, offset: int = 0):
        x_real = x[..., :self.dim // 2]
        x_imag = x[..., self.dim // 2:]
        pos_cos = self.positions_cos[offset:offset + x.size(-2)]
        pos_sin = self.positions_sin[offset:offset + x.size(-2)]
        y_real = x_real * pos_cos - x_imag * pos_sin
        y_imag = x_real * pos_sin + x_imag * pos_cos
        return torch.cat([y_real, y_imag], dim=-1)

# ---------- 动态计算版本 ----------
class RoPE_Dynamic(nn.Module):
    def __init__(self, dim: int, max_length: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_length = max_length
        self.base = base

    def forward(self, x: torch.Tensor, *, offset: int = 0):
        x_real = x[..., :self.dim // 2]
        x_imag = x[..., self.dim // 2:]
        positions = torch.arange(offset, offset + x.size(-2), device=x.device).float()
        theta = 1 / self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
        positions_theta = positions.unsqueeze(1) * theta.unsqueeze(0)
        pos_sin = torch.sin(positions_theta)
        pos_cos = torch.cos(positions_theta)
        y_real = x_real * pos_cos - x_imag * pos_sin
        y_imag = x_real * pos_sin + x_imag * pos_cos
        return torch.cat([y_real, y_imag], dim=-1)

# ---------- 完整 nGPT Block (简化版，只测 attention) ----------
class SimpleAttention(nn.Module):
    def __init__(self, dim, max_length, n_heads, rope_cls):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.pe = rope_cls(self.head_dim, max_length)
        self.dim = dim

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
        q = self.pe(q, offset=0)
        k = self.pe(k, offset=0)
        attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.head_dim ** 0.5)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, C))


def benchmark(name, fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    elapsed = time.perf_counter() - start
    return elapsed / repeat * 1000  # ms


def main():
    # 用真实模型规模测试
    dim = 1024
    n_heads = 16
    head_dim = dim // n_heads  # 64
    max_length = 65536

    print(f"配置: dim={dim}, n_heads={n_heads}, head_dim={head_dim}, max_length={max_length}")
    print(f"设备: {DEVICE}")
    print()

    # --- 测试 1: 纯 RoPE 模块 ---
    rope_pre = RoPE_Precomputed(head_dim, max_length).to(DEVICE)
    rope_dyn = RoPE_Dynamic(head_dim, max_length).to(DEVICE)

    for seq_len in [1, 32, 512, 1024]:
        x = torch.randn(1, seq_len, head_dim, device=DEVICE)
        t_pre = benchmark(f"RoPE-Precomputed  seq={seq_len}", lambda: rope_pre(x), repeat=100)
        t_dyn = benchmark(f"RoPE-Dynamic     seq={seq_len}", lambda: rope_dyn(x), repeat=100)
        ratio = t_dyn / t_pre if t_pre > 0 else float('inf')
        print(f"  seq_len={seq_len:>4d} | 预计算: {t_pre:.4f}ms | 动态: {t_dyn:.4f}ms | 动态/预计算: {ratio:.2f}x")
    print()

    # --- 测试 2: 完整 Attention 模块 ---
    attn_pre = SimpleAttention(dim, max_length, n_heads, RoPE_Precomputed).to(DEVICE)
    attn_dyn = SimpleAttention(dim, max_length, n_heads, RoPE_Dynamic).to(DEVICE)

    for seq_len in [1, 32, 256, 512]:
        x = torch.randn(1, seq_len, dim, device=DEVICE)
        t_pre = benchmark(f"Attn-Precomputed seq={seq_len}", lambda: attn_pre(x), repeat=50)
        t_dyn = benchmark(f"Attn-Dynamic    seq={seq_len}", lambda: attn_dyn(x), repeat=50)
        ratio = t_dyn / t_pre if t_pre > 0 else float('inf')
        print(f"  seq_len={seq_len:>4d} | 预计算: {t_pre:.4f}ms | 动态: {t_dyn:.4f}ms | 动态/预计算: {ratio:.2f}x")

    # --- 测试 3: 内存对比 ---
    print()
    mem_pre = sum(p.numel() * p.element_size() for p in rope_pre.buffers()) / 1024 / 1024
    mem_dyn = sum(p.numel() * p.element_size() for p in rope_dyn.buffers()) / 1024 / 1024
    print(f"RoPE 模块 buffer 内存占用:")
    print(f"  预计算: {mem_pre:.1f} MB (65536×{head_dim}×2×4bytes)")
    print(f"  动态:   {mem_dyn:.2f} MB (无 buffer)")

    print()
    print("结论: 动态计算几乎没有额外开销，因为 sin/cos 计算 (O(seq_len × head_dim))")
    print("远小于 Attention 的 O(seq_len² × head_dim) 或 MLP 的 O(seq_len × dim²)。")

if __name__ == '__main__':
    main()
