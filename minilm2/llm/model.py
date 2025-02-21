import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ( # type: ignore
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedTokenizerFast,
    AutoTokenizer
)
from transformers.modeling_outputs import CausalLMOutputWithPast # type: ignore

# nGPT
normalize = lambda x, dim=-1: F.normalize(x, p=2, dim=dim)

class NGPTConfig(PretrainedConfig):
    model_type = "ngpt"
    # 默认参数
    vocab_size: int = 32768
    dim = 768
    n_blocks = 12
    n_heads = 12
    max_position_embeddings = 1024
    dropout = .0

class MiniLM2Tokenizer(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)
    
    def _decode(self, token_ids, **kwargs):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))

class RotatoryPositionalEncoding(nn.Module):
    """旋转位置编码"""

    def __init__(self, dim: int, max_length: int):
        super().__init__()
        assert dim % 2 == 0
        positions = torch.arange(0, max_length, 1)
        theta = 1 / 10000 ** (torch.arange(0, dim, 2) / dim)  # thetai = 1/10000^(2i/dim)
        """
            theta0  theta1  theta2  theta3 ... theta(dim/2-1)
        m=0 0theta0 0theta1 0theta2 0theta3
        m=1 1theta0 1theta1 1theta2 1theta3
        m=2 2theta0 2theta1 2theta2 2theta3
        m=3 3theta0 3theta1 3theta2 3theta3
        ...
        m=max_length-1                         ...
        """
        positions_theta = positions.unsqueeze(1) * theta.unsqueeze(0)  # (max_length, dim//2)
        positions_sin = torch.sin(positions_theta)
        positions_cos = torch.cos(positions_theta)
        self.register_buffer('positions_sin', positions_sin)
        self.register_buffer('positions_cos', positions_cos)
        self.dim = dim

    def forward(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        x_real = x[..., :self.dim // 2]  # (x.size(-2), dim//2)
        x_imag = x[..., self.dim // 2:]
        pos_cos = self.positions_cos[offset:offset + x.size(-2)]  # (x.size(-2), dim//2)
        pos_sin = self.positions_sin[offset:offset + x.size(-2)]
        y_real = x_real * pos_cos - x_imag * pos_sin
        y_imag = x_real * pos_sin + x_imag * pos_cos
        return torch.cat([y_real, y_imag], dim=-1)

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.u_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # uv的缩放因子
        suinit = 1.0
        suscale = 1.0
        self.restore_scale_su = suinit / suscale
        self.su = nn.Parameter(torch.ones(hidden_dim) * suscale)
        svinit = 1.0
        svscale = 1.0
        self.restore_scale_sv = svinit / svscale
        self.sv = nn.Parameter(torch.ones(hidden_dim) * svscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_su = self.su * self.restore_scale_su
        actual_sv = self.sv * self.restore_scale_sv
        u = self.u_proj(x) * actual_su
        v = self.v_proj(x) * actual_sv * self.dim ** 0.5
        return normalize(self.o_proj(self.dropout(u * nn.functional.silu(v))))

    @torch.no_grad()
    def normalize(self) -> None:
        self.u_proj.weight.data.copy_(normalize(self.u_proj.weight.data))
        self.v_proj.weight.data.copy_(normalize(self.v_proj.weight.data))
        self.o_proj.weight.data.copy_(normalize(self.o_proj.weight.data, 0))


class CausalSelfAttention(nn.Module):
    """带因果关系的多头自注意力，使用Flash Attention和RoPE"""

    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads==0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.pe = RotatoryPositionalEncoding(self.head_dim, max_length)
        self.dropout = dropout
        self.max_length = max_length

        # QK的缩放因子
        sqkinit = 1.0
        sqkscale = 1 / dim ** 0.5
        self.restore_scale_sqk = sqkinit / sqkscale
        self.sqk = nn.Parameter(torch.ones(n_heads, 1, self.head_dim) * sqkscale)

    def forward(self, x: torch.Tensor, *, past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, T, C = x.shape
        actual_sqk = self.sqk * self.restore_scale_sqk  # (n_heads, 1, head_dim)

        # (B, T, C) -proj-> (B, T, C)
        # -view-> (B, T, n_heads, head_dim)
        # -T(1, 2)-> (B, n_heads, T, head_dim)
        if past_key_values is not None:
            k_cache, v_cache = past_key_values
            cache_size = k_cache.size(-2)
            xx = x[..., cache_size:, :]
            TT = T - cache_size
            k = self.k_proj(xx).view(B, TT, self.n_heads, -1).transpose(1, 2)
            v = self.v_proj(xx).view(B, TT, self.n_heads, -1).transpose(1, 2)
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)
        else:
            k = self.k_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
        q = self.q_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
        
        k_cache, v_cache = k.clone().detach(), v.clone().detach()

        q = self.pe(q).to(x.dtype) * actual_sqk
        k = self.pe(k).to(x.dtype) * actual_sqk

        attn_mask = torch.ones(T, k.size(-2), dtype=torch.bool, device=x.device).tril(k.size(-2) - T)

        # (B, n_heads, T, head_dim) -T(1, 2)-> (B, T, n_heads, head_dim)
        # -view-> (B, T, C)
        x = (
            nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask, dropout_p=self.dropout,
                scale=self.head_dim ** 0.5)
            .transpose(1, 2)
            .reshape(B, T, C)
        )

        return normalize(self.o_proj(x)), (k_cache, v_cache)

    @torch.no_grad()
    def clear_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def normalize(self) -> None:
        self.q_proj.weight.data.copy_(normalize(self.q_proj.weight.data))
        self.k_proj.weight.data.copy_(normalize(self.k_proj.weight.data))
        self.v_proj.weight.data.copy_(normalize(self.v_proj.weight.data))
        self.o_proj.weight.data.copy_(normalize(self.o_proj.weight.data, 0))


class NGPTBlock(nn.Module):
    """一个Decoder块"""

    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = CausalSelfAttention(dim, max_length, n_heads, dropout)
        self.mlp = MLP(dim, dim * 4, dropout)

        # 自带的学习率
        lrinit_a = 0.05
        lrscale_a = 1 / dim ** 0.5
        self.restore_scale_a = lrinit_a / lrscale_a
        self.lr_a = nn.Parameter(torch.ones(dim) * lrscale_a)
        lrinit_m = 0.05
        lrscale_m = 1 / dim ** 0.5
        self.restore_scale_m = lrinit_m / lrscale_m
        self.lr_m = nn.Parameter(torch.ones(dim) * lrscale_m)

    def forward(self, x: torch.Tensor, past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None):
        actual_lr_a = self.lr_a * self.restore_scale_a
        actual_lr_m = self.lr_m * self.restore_scale_m
        attn_out, (k, v) = self.attn(x, past_key_values=past_key_values)
        x = normalize(x + (attn_out - x) * actual_lr_a)
        x = normalize(x + (self.mlp(x) - x) * actual_lr_m)
        return x, (k, v)

    @torch.no_grad()
    def normalize(self) -> None:
        self.attn.normalize()
        self.mlp.normalize()

class NGPT(PreTrainedModel, GenerationMixin):
    """大模型本体"""
    config_class = NGPTConfig
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.blocks = nn.ModuleList([
            NGPTBlock(
                self.config.dim,
                self.config.max_position_embeddings,
                self.config.n_heads,
                self.config.dropout
            ) for _ in range(self.config.n_blocks)
        ])
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size)

        # Logit缩放因数
        szinit = 1.0
        szscale = 1 / self.config.dim ** 0.5
        self.restore_scale = szinit / szscale
        self.sz = nn.Parameter(torch.ones(self.config.vocab_size) * szscale)

        self.normalize()

    def forward(self, x: torch.Tensor,
            return_dict: bool = False,
            past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
            use_cache=False):
        B, T = x.shape
        x = self.wte(x)
        key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        cut_idx = T - self.config.max_position_embeddings
        if cut_idx > 0:
            x = x[:, cut_idx:]
            if past_key_values is not None:
                past_key_values = tuple((k[..., cut_idx:, :], v[..., cut_idx:, :]) for k, v in past_key_values)
        for i in range(self.config.n_blocks):
            block = self.blocks[i]
            if past_key_values is not None:
                x, (k, v) = block(x, past_key_values=past_key_values[i])
            else:
                x, (k, v) = block(x)
            key_values.append((k, v))
        x = self.lm_head(x)
        actual_sz = self.sz * self.restore_scale
        out = x * actual_sz
        if not return_dict:
            return out
        return CausalLMOutputWithPast(logits=out, past_key_values=tuple(key_values))

    def save(self, path: str):
        torch.save(self.state_dict(), path)  # 保存模型参数防止带上不必要的前缀

    @torch.no_grad()
    def normalize(self) -> None:
        self.wte.weight.data.copy_(normalize(self.wte.weight.data))
        self.lm_head.weight.data.copy_(normalize(self.lm_head.weight.data))
        for block in self.blocks:
            block.normalize()

    def prepare_inputs_for_generation(self, input_ids,
            past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
            use_cache=False,
            token_type_ids=None,
            attention_mask=None,
            **kwargs):
        # 准备输入，用于生成
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        return {"x": input_ids, "use_cache": use_cache, "past_key_values": past_key_values if use_cache else None}

AutoConfig.register("ngpt", NGPTConfig)
AutoModelForCausalLM.register(NGPTConfig, NGPT)
AutoTokenizer.register(NGPTConfig, fast_tokenizer_class=MiniLM2Tokenizer)

# ----------------------------------------------------------------------------
# RWKV-7: https://github.com/BlinkDL/RWKV-LM
from fla.layers import RWKV7Attention  # type: ignore

class RWKV7Config(PretrainedConfig):
    model_type = "rwkv7"
    # 默认参数
    vocab_size: int = 32768
    dim = 768
    n_blocks = 12
    max_length = 1024
    dropout = .0
    max_lr = 1e-4

class TMix(nn.Module):
    def __init__(self, dim: int, block_id: int, n_blocks: int):
        super().__init__()
        # 根据BlinkDL原版实现中的建议取值
        decay_low_rank_dim = max(32, int(round(1.8 * (dim ** 0.5) / 32) * 32))
        a_low_rank_dim     = max(32, int(round(1.8 * (dim ** 0.5) / 32) * 32))
        v_low_rank_dim     = max(32, int(round(1.3 * (dim ** 0.5) / 32) * 32))
        gate_low_rank_dim  = max(32, int(round(0.6 * (dim ** 0.8) / 32) * 32))
        self.rwkv7 = RWKV7Attention(
            "chunk",
            dim,
            layer_idx=block_id,
            decay_low_rank_dim=decay_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim
        )
        with torch.no_grad():  # 参数初始化，从BlinkDL原版实现中复制过来的
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

            ratio_1_to_0 = 1-(block_id / n_blocks)
            ratio_0_to_1 = block_id / (n_blocks-1)
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
                decay_speed[n] = - 7 + 5 * (n / (dim-1)) ** (0.85+1.0 * ratio_0_to_1 ** 0.5)  # WTF?
            w0 = decay_speed+0.5
            self.rwkv7.w_lora.lora[0].weight.data.copy_(w1.T)
            self.rwkv7.w_lora.lora[2].weight.data.copy_(w2.T)
            self.rwkv7.w_lora.lora[2].bias.data.copy_(w0)

            a1 = torch.zeros(dim, a_low_rank_dim)
            a2 = ortho_init(torch.zeros(a_low_rank_dim, dim), 0.1)
            a0 = torch.zeros(dim)
            self.rwkv7.a_lora.lora[0].weight.data.copy_(a1.T)
            self.rwkv7.a_lora.lora[2].weight.data.copy_(a2.T)
            self.rwkv7.a_lora.lora[2].bias.data.copy_(a0)

            if block_id != 0:  # 第一层没有这个模块
                v1 = torch.zeros(dim, v_low_rank_dim)
                v2 = ortho_init(torch.zeros(v_low_rank_dim, dim), 0.1)
                v0 = torch.zeros(dim)+1.0
                self.rwkv7.v_lora.lora[0].weight.data.copy_(v1.T)
                self.rwkv7.v_lora.lora[2].weight.data.copy_(v2.T)
                self.rwkv7.v_lora.lora[2].bias.data.copy_(v0)
                del v1, v2, v0

            g1 = torch.zeros(dim, gate_low_rank_dim)
            g2 = ortho_init(torch.zeros(gate_low_rank_dim, dim), 0.1)
            self.rwkv7.g_lora.lora[0].weight.data.copy_(g1.T)
            self.rwkv7.g_lora.lora[2].weight.data.copy_(g2.T)

            self.rwkv7.r_proj.weight.data.uniform_(-0.5 / (dim ** 0.5), 0.5 / (dim ** 0.5))
            self.rwkv7.k_proj.weight.data.uniform_(-0.05 / (dim ** 0.5), 0.05 / (dim ** 0.5))
            self.rwkv7.v_proj.weight.data.uniform_(-0.5 / (dim ** 0.5), 0.5 / (dim ** 0.5))
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
            ratio_1_to_0 = 1-(block_id / n_blocks)
            # ddd = torch.linspace(0, 1, dim) * (dim - 1) / dim
            ddd = torch.ones(dim)
            for i in range(dim):
                ddd[i] = i / dim
            self.x_k = nn.Parameter(1.0-torch.pow(ddd, ratio_1_to_0 ** 4))
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)

        self.key.weight.data.uniform_(-0.5 / (dim ** 0.5), 0.5 / (dim ** 0.5))
        self.value.weight.data.zero_()

        del ddd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.shift1(x)-x
        k = x+xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class RWKV7Block(nn.Module):
    """一个Decoder块"""
    def __init__(self, dim: int, block_id: int, n_blocks: int):
        super().__init__()
        self.attn = TMix(dim, block_id, n_blocks)
        self.mlp = CMix(dim, dim * 4, block_id, n_blocks)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x_attn, v_first = self.attn(self.norm1(x), v_first=v_first)
        x = x+x_attn
        x = x+self.mlp(self.norm2(x))
        assert v_first is not None, "v_first should not be None"
        return x, v_first


class RWKV7(PreTrainedModel, GenerationMixin):
    """大模型本体"""
    config_class = RWKV7Config
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        assert self.config.dim % 64 == 0, "dim必须是64的倍数"
        assert math.log2(self.config.vocab_size).is_integer(), "vocab_size必须是2的幂"
        self.wte = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.blocks = nn.ModuleList([
            RWKV7Block(self.config.dim, i, self.config.n_blocks)
            for i in range(self.config.n_blocks)
        ])
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size)
        self.norm_in = nn.LayerNorm(self.config.dim)
        self.norm_out = nn.LayerNorm(self.config.dim)
        self.wte.weight.data.uniform_(-self.config.max_lr, self.config.max_lr)
        nn.init.orthogonal_(self.lm_head.weight, gain=0.5)

    def forward(self, x: torch.Tensor):
        x = self.norm_in(self.wte(x))
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first)
        return self.lm_head(self.norm_out(x))

    def save(self, path: str):
        torch.save(self.state_dict(), path)  # 保存模型参数防止带上不必要的前缀

AutoConfig.register("rwkv7", RWKV7Config)
AutoModelForCausalLM.register(RWKV7Config, RWKV7)
