import math
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from .model import *
from .dataset import PreTrainDataset, collate_fn, from_file
from .validate import validate
from . import config
from .lr_schedule import get_lr_schedule

if __name__ == '__main__':
    import sys
    import os
    import json
    
    if len(sys.argv) < 2:
        print('Usage: python -m minilm2.llm.train <config_path>')
        exit(1)
    config_path = sys.argv[1]
    config_dir = os.path.dirname(config_path) # 配置文件路径
    train_config = json.load(open(config_path))

    # 根据配置文件创建模型
    model_type = train_config["model"]
    print(f"Loading {model_type} model...")

    # 加载tokenizer并获取词表大小
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(config_dir, train_config['tokenizer_path']))
    vocab_size = len(tokenizer)
    print(f"==> Vocab size: {vocab_size}")

    # 根据配置文件创建模型
    model_type = train_config["model"]
    print(f"Loading {model_type} model...")
    if model_type == "NGPT":
        model_config = NGPTConfig(
            vocab_size=vocab_size,
            dim=train_config["model_dim"],
            n_blocks=train_config["num_layers"],
            n_heads=train_config["num_heads"],
            max_position_embeddings=train_config["max_length"],
            dropout=train_config["dropout"]
        )
    elif model_type == "RWKV7":
        model_config = RWKV7Config(
            vocab_size=2 ** math.ceil(math.log2(vocab_size)),
            dim=train_config["model_dim"],
            n_blocks=train_config["num_layers"],
            max_position_embeddings=train_config["max_length"],
            dropout=train_config["dropout"],
            max_lr=train_config["max_learning_rate"]
        )
    model = (AutoModelForCausalLM.from_pretrained(os.path.join(config_dir, train_config['model_path']))
                if train_config['model_path'] else
                AutoModelForCausalLM.from_config(model_config))
    # 统计参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"==> Number of parameters: {params / 1e6:.2f}M")

    # 将模型移动到显存并编译以加速训练
    model.to(config.DEVICE)
    print("==> Compiling model...")
    model.compile()
    model.train()

    # 加载数据集
    print("Loading dataset...")
    train_dataset, val_dataset = from_file(
        os.path.join(config_dir, train_config["dataset_path"]),
        train_config["max_length"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS
    )

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), fused=True, weight_decay=0.0)

    # 定义学习率衰减策略
    lr_schedule = get_lr_schedule(
        train_config["max_learning_rate"],
        train_config["min_learning_rate"],
        train_config["warmup_steps"],
        train_config["total_steps"]
    )

    micro_step = 0
    step = train_config['checkpoint_step']
    total_loss = 0.0
    print("Start training...")
    log_fname = os.path.join(config_dir, train_config['log_file'])
    print(f"==> Log file: {log_fname}")
    torch.set_float32_matmul_precision('high') # 调整精度以加速训练
    try:
        with tqdm(train_loader) as pbar:
            for x, y in pbar:
                # 一个step的开始，更新学习率
                if micro_step % train_config["n_batches_per_step"] == 0:
                    optimizer.zero_grad()
                    lr = lr_schedule(step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                micro_step += 1

                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction='mean',
                    ignore_index=config.SPECIAL_TOKENS["<pad>"]
                ) / train_config['n_batches_per_step']
                del x, y, logits # 释放显存
                loss.backward() # 反向传播积累梯度
                total_loss += loss.item()
                pbar.set_description(f'loss: {loss.item() * train_config["n_batches_per_step"]:.4f} lr: {lr:.4f}')

                # 一个step的结束，更新参数并保存日志
                if micro_step % train_config['n_batches_per_step'] == 0:
                    step += 1
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if model_type == "NGPT":
                        model.normalize() # NGPT需要在每个训练步进行参数归一化
                    open(log_fname, 'a').write(f'TRAIN,{step},{lr},{total_loss}\n')
                    total_loss = 0.0

                    # 定期进行验证并保存检查点
                    if step % train_config["validation_interval"] == 0:
                        print('==> Validating...')
                        model.eval() # 切换到验证模式
                        val_loss = validate(model, val_dataset, train_config["val_batch_size"])
                        model.train() # 在验证时切换到了推理模式，切换回来
                        print(f'==> Validation loss: {val_loss:.4f}')
                        checkpoint_name = f'checkpoint_{step}_{val_loss:.4f}.pt'
                        model.save_pretrained(os.path.join(config_dir, checkpoint_name))
                        print(f'==> Saved checkpoint to {checkpoint_name}')
                        open(log_fname, 'a').write(f'VAL,{step},{lr},{val_loss}\n')

    except KeyboardInterrupt:
        print('Training interrupted.')
        # 保存未使用的数据集
        assert isinstance(train_loader.dataset, PreTrainDataset)
        used_indexes = train_loader.dataset.get_used_indexes()
        dataset_path = os.path.dirname(os.path.join(config_dir, train_config['dataset_path']))
        lst_name = os.path.join(dataset_path, f'train{step}_used.lst')
        with open(lst_name, 'w') as f:
            for i in tqdm(used_indexes):
                f.write(f'{i}\n')
        print(f"==> Unused indexes saved to {lst_name}")
        print("!! REMEMBER TO UPDATE THE DATASET FILE AND CONFIG FILE TO USE THE UPDATED LIST AND CHECKPOINT !!")

    finally:
        # 保存最终的检查点
        checkpoint_name = f'checkpoint_{step}.pt'
        model.save_pretrained(os.path.join(config_dir, checkpoint_name))
        print(f'==> Saved checkpoint to {checkpoint_name}')
