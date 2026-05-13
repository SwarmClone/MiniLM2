from fla.modules import LayerNorm
from torch.nn.modules.container import ModuleList
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from fla.layers import KimiDeltaAttention
from fla.models.utils import Cache, FLAGenerationMixin

class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
    # 默认参数
    vocab_size: int = 32768
    dim = 768
    num_hidden_layers = 12
    n_heads = 12
    dropout = .0

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

class GLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.u_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.u_proj(x)
        v = self.v_proj(x)
        return self.o_proj(self.dropout(u * nn.functional.silu(v)))

class KimiLinearBlock(nn.Module):
    """一个Decoder块"""

    def __init__(self, dim: int, n_heads: int, dropout: float, layer_idx: int):
        super().__init__()
        self.attn = KimiDeltaAttention(hidden_size=dim, num_heads=n_heads, dropout=dropout, layer_idx=layer_idx)
        self.mlp = GLU(dim, dim * 4, dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, x: torch.Tensor, past_key_values: Cache | None = None):
        attn_out, _, past_key_values = self.attn(self.norm1(x), past_key_values=past_key_values)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, past_key_values

class KimiLinear(PreTrainedModel, FLAGenerationMixin):
    """大模型本体"""
    config_class = KimiLinearConfig
    all_tied_weights_keys = {
        "lm_head.weight": "wte.weight",
    }
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.blocks: ModuleList = nn.ModuleList([
            KimiLinearBlock(
                self.config.dim,
                self.config.n_heads,
                self.config.dropout,
                layer_idx=i
            ) for i in range(self.config.num_hidden_layers)
        ])
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)

    @classmethod
    def _prepare_cache_for_generation(cls, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length):
        if not generation_config.use_cache:
            return
        if model_kwargs.get("past_key_values") is not None:
            return
        model_kwargs["past_key_values"] = Cache()

    def forward(self, input_ids: torch.Tensor,
            return_dict: bool = False,
            past_key_values: Cache | None = None,
            use_cache: bool = False, **kwargs):
        x = self.wte(input_ids)
        for block in self.blocks:
            if not use_cache:
                x, _ = block(x)
            else:
                x, past_key_values = block(x, past_key_values=past_key_values)
            
        x = self.lm_head(x)
        out = x
        if not return_dict:
            return out
        if use_cache:
            return CausalLMOutputWithPast(logits=out, past_key_values=past_key_values)
        return CausalLMOutput(logits=out)

AutoConfig.register("kimi_linear", KimiLinearConfig)
AutoModelForCausalLM.register(KimiLinearConfig, KimiLinear)
AutoTokenizer.register(KimiLinearConfig, fast_tokenizer_class=MiniLM2Tokenizer)
