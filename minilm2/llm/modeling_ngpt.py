from torch.nn.modules.container import ModuleList
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache, DynamicCache

# nGPT
normalize = lambda x, dim=-1: F.normalize(x, p=2, dim=dim)

class NGPTConfig(PretrainedConfig):
    model_type = "ngpt"
    # 默认参数
    vocab_size: int = 32768
    dim = 768
    num_hidden_layers = 12
    n_heads = 12
    max_position_embeddings = 1024
    dropout = .0
    rope_base = 10000

class MiniLM2Tokenizer(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_tokens_to_string(self, tokens: list[str]):
        return ''.join(tokens)
    
    def _decode(
            self,
            token_ids: list[int] | int,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool | None = None,
            **kwargs):
        tokens = self.convert_ids_to_tokens(token_ids)
        if isinstance(tokens, str):
            tokens = [tokens]
        return self.convert_tokens_to_string(tokens)

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码"""

    def __init__(self, dim: int, max_length: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_length = max_length
        self.base = base

    def _compute_sin_cos(self, device: torch.device, offset: int, length: int):
        positions = torch.arange(offset, offset + length, device=device).float()
        theta = 1 / self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        positions_theta = positions.unsqueeze(1) * theta.unsqueeze(0)
        return torch.sin(positions_theta), torch.cos(positions_theta)

    def forward(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        x_real = x[..., :self.dim // 2]
        x_imag = x[..., self.dim // 2:]
        pos_sin, pos_cos = self._compute_sin_cos(x.device, offset, x.size(-2))
        y_real = x_real * pos_cos - x_imag * pos_sin
        y_imag = x_real * pos_sin + x_imag * pos_cos
        return torch.cat([y_real, y_imag], dim=-1)

class NormalizedMLP(nn.Module):
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

class NormalizedCausalSelfAttention(nn.Module):
    """带因果关系的多头自注意力，使用Flash Attention和RoPE"""

    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float, rope_base: int = 10000):
        super().__init__()
        assert dim % n_heads==0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.pe = RotaryPositionEmbedding(self.head_dim, max_length, rope_base)
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
            new_k = self.k_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
            new_v = self.v_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
            k = torch.cat([k_cache, new_k], dim=-2)
            v = torch.cat([v_cache, new_v], dim=-2)
            q = self.q_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
        else:
            cache_size = 0
            new_k = self.k_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
            new_v = self.v_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
            k = new_k
            v = new_v
            q = self.q_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)

        q = self.pe(q, offset=cache_size).to(x.dtype) * actual_sqk
        k = self.pe(k).to(x.dtype) * actual_sqk

        attn_mask = torch.ones(T, T + cache_size, dtype=torch.bool, device=x.device).tril(cache_size)

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

        return normalize(self.o_proj(x)), (new_k, new_v)

    @torch.no_grad()
    def normalize(self) -> None:
        self.q_proj.weight.data.copy_(normalize(self.q_proj.weight.data))
        self.k_proj.weight.data.copy_(normalize(self.k_proj.weight.data))
        self.v_proj.weight.data.copy_(normalize(self.v_proj.weight.data))
        self.o_proj.weight.data.copy_(normalize(self.o_proj.weight.data, 0))

class NGPTBlock(nn.Module):
    """一个Decoder块"""

    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float, rope_base: int = 10000):
        super().__init__()
        self.attn = NormalizedCausalSelfAttention(dim, max_length, n_heads, dropout, rope_base)
        self.mlp = NormalizedMLP(dim, dim * 4, dropout)

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
        xx = x[..., -attn_out.size(-2):, :]
        xx = normalize(xx + (attn_out - xx) * actual_lr_a)
        xx = normalize(xx + (self.mlp(xx) - xx) * actual_lr_m)
        return F.pad(xx, (0, 0, x.size(-2) - xx.size(-2), 0)), (k, v)

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
        self.blocks: ModuleList = nn.ModuleList([
            NGPTBlock(
                self.config.dim,
                self.config.max_position_embeddings,
                self.config.n_heads,
                self.config.dropout,
                self.config.rope_base
            ) for _ in range(self.config.num_hidden_layers)
        ])
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size)

        self.all_tied_weights_keys = {}

        # Logit缩放因数
        szinit = 1.0
        szscale = 1 / self.config.dim ** 0.5
        self.restore_scale = szinit / szscale
        self.sz = nn.Parameter(torch.ones(self.config.vocab_size) * szscale)

        self.normalize()

    def forward(self, input_ids: torch.Tensor,
            return_dict: bool = False,
            past_key_values: DynamicCache | None = None,
            use_cache: bool = False, **kwargs):
        x = self.wte(input_ids)
        empty_cache = False
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
                empty_cache = True
            elif past_key_values.get_seq_length() == 0:
                empty_cache = True
        for i in range(self.config.num_hidden_layers):
            block = self.blocks[i]
            if empty_cache or not use_cache:
                x, (k, v) = block(x)
            else:
                layer = past_key_values.layers[i]
                x, (k, v) = block(x, past_key_values=(layer.keys, layer.values))
            if use_cache:
                past_key_values.update(k, v, i)
            
        x = self.lm_head(x)
        actual_sz = self.sz * self.restore_scale
        out = x * actual_sz
        if not return_dict:
            return out
        if use_cache:
            return CausalLMOutputWithPast(logits=out, past_key_values=past_key_values)
        return CausalLMOutput(logits=out)

    @torch.no_grad()
    def normalize(self) -> None:
        self.wte.weight.data.copy_(normalize(self.wte.weight.data))
        self.lm_head.weight.data.copy_(normalize(self.lm_head.weight.data))
        for block in self.blocks:
            block.normalize()  # pyright: ignore[reportCallIssue]
    

AutoConfig.register("ngpt", NGPTConfig)
AutoModelForCausalLM.register(NGPTConfig, NGPT)
AutoTokenizer.register(NGPTConfig, fast_tokenizer_class=MiniLM2Tokenizer)
