import math
import torch
from tqdm import tqdm
from torch.nn import functional as F
from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteriaList,
    StopStringCriteria,
    EosTokenCriteria
)
from .model import *
from . import config

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
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(config_dir, train_config['tokenizer_path']))
    vocab_size = len(tokenizer)
    print(f"==> Vocab size: {vocab_size}")

    # 根据配置文件创建模型
    model_type = train_config["model"]
    print(f"Loading {model_type} model...")
    model = AutoModelForCausalLM.from_pretrained(os.path.join(config_dir, train_config['model_path']))
    # 统计参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"==> Number of parameters: {params / 1e6:.2f}M")

    # 将模型移动到显存并编译以加速推理
    model.to(config.DEVICE)
    model.eval()

    torch.set_float32_matmul_precision('high') # 调整精度以加速推理
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
            break
        # 编码输入文本
        inputs = tokenizer([text], return_tensors='pt').to(config.DEVICE)
        # 进行流式推理
        with torch.no_grad():
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            try:
                model.generate(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True)
            except KeyboardInterrupt:
                pass
