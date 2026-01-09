import pathlib
import math
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from torch import nn, optim
from flash_muon import Muon
from typing import Any
from minilm2.llm.modeling_ngpt import NGPTConfig

def get_optimizers(model: nn.Module, train_config: dict[str, Any]) -> list[tuple[optim.Optimizer, float]]:
    # 接受模型和训练配置，返回优化器列表
    optimizers_with_lrscale: list[tuple[optim.Optimizer, float]] = []
    match train_config['optimizer']:
        case 'adamw':
            optimizers_with_lrscale.append((
                optim.AdamW(
                    model.parameters(),
                    fused=True,
                    betas=tuple(train_config['betas']),
                    weight_decay=train_config['weight_decay']
                ),
                1.0
        ))
        case 'muon':
            muon_params_dict = {
                n: p for n, p in model.named_parameters()
                if p.ndim == 2 and 'wte' not in n and 'lm_head' not in n
            }
            optimizers_with_lrscale.append((
                Muon(
                    params=muon_params_dict.values(),
                    weight_decay=train_config['weight_decay']
                ),
                1.0
            ))
            adam_params_dict = {
                n: p for n, p in model.named_parameters()
                if n not in muon_params_dict
            }
            optimizers_with_lrscale.append((
                optim.AdamW(
                    adam_params_dict.values(),
                    weight_decay=train_config['weight_decay'],
                    betas=tuple(train_config['betas']),
                    fused=True
                ),
                1.0
            ))
        case _:
            raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")
    
    return optimizers_with_lrscale

def get_model_tokenizer(config_dir: pathlib.Path, train_config: dict[str, Any]) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    # 接受训练配置，返回模型与分词器实例

    # 1. 分词器
    tokenizer = AutoTokenizer.from_pretrained(config_dir / train_config["tokenizer_path"])
    vocab_size = len(tokenizer)

    # 2. 模型
    if train_config['model_path']: # 已有模型检查点，从检查点加载
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(config_dir / train_config["model_path"])
        return (model, tokenizer)
    # 无检查点，从配置创建新模型
    model_type = train_config["model"]
    match model_type.lower():
        case "ngpt":
            model_config = NGPTConfig(
                vocab_size=2 ** math.ceil(math.log2(vocab_size)),
                dim=train_config["model_dim"],
                n_blocks=train_config["num_layers"],
                n_heads=train_config["num_heads"],
                max_position_embeddings=train_config["max_length"],
                dropout=train_config["dropout"],
                rope_base=train_config["rope_base"]
            )
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
    model: PreTrainedModel = AutoModelForCausalLM.from_config(model_config)
    return (model, tokenizer)

def load_optimizer_states(
        config_dir: pathlib.Path,
        train_config: dict[str, Any],
        optimizers_with_lrscale: list[tuple[optim.Optimizer, float]]
        ) -> list[tuple[optim.Optimizer, float]]:
    # 加载优化器状态
    optimizer_state_path = config_dir / train_config['optimizer_state_path']
    print(f"==> Loading optimizer state from {optimizer_state_path}")
    state_dict_all = torch.load(optimizer_state_path, weights_only=True)
    for i, state_dict in state_dict_all.items():
        optimizers_with_lrscale[i][0].load_state_dict(state_dict)
    return optimizers_with_lrscale

__all__ = [
    "get_optimizers",
    "get_model_tokenizer",
    "load_optimizer_states"
]
