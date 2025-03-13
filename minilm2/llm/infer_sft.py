import math
import torch
from tqdm import tqdm
from torch.nn import functional as F
from .model import NGPT, RWKV7
from . import config
from tokenizers import Tokenizer # type: ignore

def build_context(history: list[tuple[str, str]], tokenizer: Tokenizer,
                max_length: int, system_prompt: str | None = None) -> torch.Tensor:
    ids = []
    human_prefix_ids = tokenizer.encode(config.HUMAN_PREFIX).ids
    ai_prefix_ids = tokenizer.encode(config.AI_PREFIX).ids
    separator_ids = tokenizer.encode("\n" * 3).ids
    system_prompt_ids = tokenizer.encode(system_prompt).ids + separator_ids if system_prompt else []
    for i in range(len(history)):
        turn = history[i]
        ids += human_prefix_ids + tokenizer.encode(turn[0]).ids + separator_ids
        ids += ai_prefix_ids + tokenizer.encode(turn[1]).ids
        if i < len(history) - 1:
            ids += separator_ids
    ids = system_prompt_ids + ids[len(system_prompt_ids)-max_length:]
    return torch.LongTensor(ids).unsqueeze(0)

def append_history(history: list[tuple[str, str]], role: str, text: str) -> list[tuple[str, str]]:
    if role == "human":
        history.append((text, ""))
    else:
        history[-1] = (history[-1][0], text)
    return history

def load_in_fp16(model, model_path=None):
    if not model_path:
        ckpt = os.path.join(config_dir, train_config["checkpoint_file"])
    else:
        ckpt = model_path
    
    state_dict = torch.load(ckpt)
    for name, param in model.named_parameters():
        if name in state_dict.keys():
            param.data = state_dict[name].to(torch.float16)
            
    model.dtype = torch.float16
    return model

if __name__ == '__main__':
    import sys
    import os
    import json
    if len(sys.argv) < 2:
        print('Usage: python -m minilm2.llm.infer_sft <config_path>')
        exit(1)
    config_path = sys.argv[1]
    config_dir = os.path.dirname(config_path) # 配置文件路径
    train_config = json.load(open(config_path))

    # 加载tokenizer并获取词表大小
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(os.path.join(config_dir, train_config['tokenizer_path']))
    vocab_size = tokenizer.get_vocab_size()
    print(f"==> Vocab size: {vocab_size}")

    # 根据配置文件创建模型
    model_type = train_config["model"]
    print(f"Loading {model_type} model...")
    model: torch.nn.Module
    if model_type == "NGPT":
        model = NGPT(
            vocab_size=vocab_size,
            dim=train_config['model_dim'],
            max_length=train_config['max_length'],
            n_heads=train_config['num_heads'],
            n_blocks=train_config['num_layers'],
            dropout=0 # 推理时不使用dropout
        )
    elif model_type == "RWKV7":
        model = RWKV7(
            vocab_size=2 ** math.ceil(math.log2(vocab_size)), # 确保vocab_size为2的幂
            dim=train_config['model_dim'],
            n_blocks=train_config['num_layers'],
            max_lr=train_config['max_learning_rate']
        )
    # 统计参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"==> Number of parameters: {params / 1e6:.2f}M")
    # 加载已有的检查点
    if train_config['checkpoint_file']:
        checkpoint_path = os.path.join(config_dir, train_config['checkpoint_file'])
        print(f"==> Loading checkpoint from {checkpoint_path}, step={train_config['checkpoint_step']}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    model = load_in_fp16(model, checkpoint_path)

    # 将模型移动到显存并编译以加速推理
    print(f"==> Using device: {config.DEVICE}")
    torch.set_float32_matmul_precision('high') # 调整精度以加速推理
    model.to(config.DEVICE)
    print(f"==> Compiling model...")
    model.compile(fullgraph=True)
    model.eval()

    history: list[tuple[str, str]] = []

    if config.ENABLE_KVCACHE:
        print("==> KV Cache enabled.")
        sysprompt = build_context([], tokenizer, train_config['max_length'], train_config.get("system_prompt")).to(config.DEVICE)
        model.update(sysprompt) # 在缓存中加入初始输入
    print("Use '!clear' to clear history. Use '!top_p x.xx' to adjust top_p. Use '!temperature x.xx' to adjust temperature.")
    while True:
        text = ""
        try:
            while True:
                if not text:
                    text = input("Use '\\' to enter multi-line input. Press Ctrl-D to quit. > ")
                else:
                    text += input("> ")
                if text[-1] != '\\':
                    break
                text = text[:-1] + "\n"
        except EOFError:
            print()
            break
        if text == "!clear":
            history = []
            if config.ENABLE_KVCACHE:
                model.clear_cache()
                sysprompt = build_context([], tokenizer, train_config['max_length'], train_config.get("system_prompt")).to(config.DEVICE)
                model.update(sysprompt) # 在缓存中加入初始输入
            print("History cleared.")
            continue
        if text.startswith("!top_p"):
            try:
                top_p = float(text.split()[1])
                if top_p <= 0 or top_p > 1:
                    raise ValueError
                train_config['top_p'] = top_p
                print(f"top_p set to {top_p}.")
            except (ValueError, IndexError):
                print("top_p value should be a float between 0 and 1. Usage: !top_p x.xx")
            continue
        if text.startswith("!temperature"):
            try:
                temperature = float(text.split()[1])
                if temperature <= 0 or temperature > 2:
                    raise ValueError
                train_config['temperature'] = temperature
                print(f"temperature set to {temperature}.")
            except (ValueError, IndexError):
                print("temperature value should be a float between 0 and 2. Usage: !temperature x.xx")
            continue
        if text.startswith("!context"):
            for i in build_context(
                history,
                tokenizer,
                train_config['max_length'],
                system_prompt=train_config.get("system_prompt")).squeeze():
                print(tokenizer.id_to_token(i.item()), end="", flush=True)
            print()
            continue
        if text.startswith("!history"):
            print(history)
            continue
        # 加入历史记录
        history = append_history(history, "human", text)
        # 构建输入
        input_ids = build_context(history, tokenizer, train_config['max_length'], train_config.get("system_prompt"))
        input_ids = input_ids.to(config.DEVICE)
        # 推理
        response = ""
        n_blankline = 0
        with torch.no_grad():
            while not config.ENABLE_KVCACHE:
                try:
                    output = model(input_ids)
                    logits = F.softmax(output[0][-1] / train_config['temperature'], dim=-1)
                    # 采样输出，取概率最高的n个进行加权随机采样
                    probs, indices = logits.topk(round(vocab_size * train_config['top_p']))
                    sample = torch.multinomial(probs, 1)
                    token_id = indices[sample]
                    prob = probs[sample].item()
                    confidence_level = round(prob ** 0.5 * 16) # 开方以增大低概率时的颜色差异
                    input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=1)[:, -train_config['max_length']:] # 自回归生成
                    token = tokenizer.id_to_token(token_id.item())
                    if token == "\n":
                        n_blankline += 1
                        if n_blankline >= 3:
                            break
                    else:
                        n_blankline = 0
                    print(f"\033[1;38;5;{confidence_level + 239}m{token}\033[0m", end="", flush=True)
                    response += token
                except KeyboardInterrupt:
                    print("\033[0m")
                    break
            
            last_out: torch.Tensor | None = None
            while config.ENABLE_KVCACHE:
                try:
                    if not last_out:
                        input_ids = build_context(history[-1:], tokenizer, train_config['max_length']).to(config.DEVICE)
                        last_out_logits = model.update(input_ids)[:, -1, :] # 更新缓存
                    else:
                        last_out_logits = model.update(last_out)[:, -1, :] # 更新缓存
                    logits = F.softmax(last_out_logits[0] / train_config['temperature'], dim=-1)
                    # 采样输出，取概率最高的n个进行加权随机采样
                    probs, indices = logits.topk(round(vocab_size * train_config['top_p']))
                    sample = torch.multinomial(probs, 1)
                    token_id = indices[sample]
                    last_out = token_id.unsqueeze(0)
                    prob = probs[sample].item()
                    confidence_level = round(prob ** 0.5 * 16) # 开方以增大低概率时的颜色差异
                    token = tokenizer.id_to_token(token_id.item())
                    if token == "\n":
                        n_blankline += 1
                        if n_blankline >= 3:
                            break
                    else:
                        n_blankline = 0
                    print(f"\033[1;38;5;{confidence_level + 239}m{token}\033[0m", end="", flush=True)
                    response += token
                except KeyboardInterrupt:
                    print("\033[0m")
                    break
        # 加入历史记录
        history = append_history(history, "ai", response.strip())
