import warnings
import time
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import *
from .dataset_sft import SFTDataset, collate_fn, from_file
from . import config
from .lr_schedule import get_lr_schedule
from .muon import Muon

if __name__ == '__main__':
    import sys
    import os
    import json
    
    if len(sys.argv) < 2:
        print('Usage: python -m minilm2.llm.sft <config_path>')
        exit(1)
    train_config = json.load(open("models/defaults.json"))
    config_path = sys.argv[1]
    config_dir = os.path.dirname(config_path) # 配置文件路径
    train_config.update(json.load(open(config_path)))

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
    params = sum(p.numel() for p in model.parameters())
    print(f"==> Number of parameters: {params / 1e6:.2f}M")

    # 去除不需要的梯度
    if 'finetune_layers' in train_config and (n_freezed_layers := train_config['num_layers'] - train_config['finetune_layers']) > 0:
        print(f"==> Freezing the first {n_freezed_layers} layers...")
        for block in model.blocks[:n_freezed_layers]:
            for param in block.parameters():
                param.requires_grad = False

    # 将模型移动到显存并编译以加速训练
    model.to(config.DEVICE)
    scaler = GradScaler(enabled=train_config['bfloat16']) # 如果启用bfloat16则启用混合精度训练
    print("==> Compiling model...")
    # model.compile()
    model.train()

    # 是否启用梯度检查点
    checkpointing_kwargs: dict[str, bool | int] = {}
    if train_config.get("gradient_checkpointing"):
        if train_config.get("model").lower() == "ngpt":
            checkpointing_kwargs = {
                "checkpointing": True,
                "checkpointing_segments": train_config.get("gradient_checkpointing_segments", 1)
            }
        else:
            print("!! 非NGPT模型暂不支持使用梯度检查点 !!")

    # 加载数据集
    print("Loading dataset...")
    train_dataset = from_file(
        os.path.join(config_dir, train_config["dataset_path"]),
        train_config["max_length"])
    if config.NUM_WORKERS != 0: # 如果需要正常保存数据使用状态，则必须是0
        warnings.warn((
            f"Using {config.NUM_WORKERS} workers for data loading."
            "Might not be able to save the usage properly."
            "Consider setting `config.NUM_WORKERS = 0`."
        ))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS
    )

    # 定义优化器
    optimizers_with_lrscale: list[tuple[optim.Optimizer, float]] = []
    if train_config['optimizer'] == 'adamw':
        optimizers_with_lrscale.append((
            optim.AdamW(
                model.parameters(),
                fused=True,
                betas=tuple(train_config['betas']),
                weight_decay=train_config['weight_decay']
            ),
            1.0
        ))
    elif train_config['optimizer'] == 'muon':
        muon_params_dict = {
            n: p for n, p in model.named_parameters()
            if p.ndim == 2 and 'wte' not in n and 'lm_head' not in n
        }
        muon_params_dict_2x = {
            n: p for n, p in muon_params_dict.items()
            if 'w_lora' in n
        }
        muon_params_dict_1x = {
            n: p for n, p in muon_params_dict.items()
            if 'w_lora' not in n
        }
        adam_params_dict = {
            n: p for n, p in model.named_parameters()
            if n not in muon_params_dict
        }
        optimizers_with_lrscale.append((
            Muon(
                muon_params=muon_params_dict_1x.values(),
                wd=train_config['weight_decay'],
                adamw_betas=tuple(train_config['betas'])
            ),
            1.0
        ))
        optimizers_with_lrscale.append((
            Muon(
                muon_params=muon_params_dict_2x.values(),
                wd=train_config['weight_decay'],
                adamw_betas=tuple(train_config['betas'])
            ),
            2.0
        ))
        optimizers_with_lrscale.append((
            optim.AdamW(
                adam_params_dict.values(),
                weight_decay=train_config['weight_decay'],
                betas=tuple(train_config['betas']),
                fused=True
            ),
            1.0
        ))
    else:
        raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")
    # 如果有的话，加载优化器状态
    if train_config['optimizer_state_path']:
        optimizer_state_path = os.path.join(config_dir, train_config['optimizer_state_path'])
        print(f"==> Loading optimizer state from {optimizer_state_path}")
        state_dict_all = torch.load(optimizer_state_path, weights_only=True)
        for i, state_dict in state_dict_all.items():
            optimizers_with_lrscale[i][0].load_state_dict(state_dict)

    # 定义学习率衰减策略
    lr_schedule = get_lr_schedule(
        train_config["max_learning_rate"],
        train_config["min_learning_rate"],
        train_config["warmup_steps"],
        train_config["total_steps"]
    )

    micro_step = 0
    lr = 0
    step = train_config['checkpoint_step']
    total_loss = 0.0
    print("Start training...")
    log_fname = os.path.join(config_dir, train_config['log_file'])
    print(f"==> Log file: {log_fname}")
    torch.set_float32_matmul_precision('high') # 调整精度以加速训练
    try:
        with tqdm(train_loader) as pbar:
            for x, y, m in pbar:
                if m.sum() == 0:
                    continue # 防止除0导致NaN

                # 一个step的开始，更新学习率
                if micro_step % train_config["n_batches_per_step"] == 0:
                    total_loss = 0.0
                    for optimizer, lr_scale in optimizers_with_lrscale:
                        optimizer.zero_grad()
                        lr = lr_schedule(step)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr * lr_scale
                micro_step += 1

                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)
                m = m.to(config.DEVICE)
                with autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=train_config["bfloat16"] # 启用bfloat16则使用混合精度训练
                    ):
                    logits = model(x, **checkpointing_kwargs)
                    loss = (F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        reduction="none",
                        ignore_index=config.SPECIAL_TOKENS["<pad>"]
                    ) * m.view(-1)).sum() / m.sum() / train_config['n_batches_per_step']
                    del x, y, m, logits # 释放显存
                scaler.scale(loss).backward() # 反向传播积累梯度，使用缩放来避免精度损失
                total_loss += loss.item()
                pbar.set_description(f'loss: {loss.item() * train_config["n_batches_per_step"]:.4f} lr: {lr:.4f}')

                # 一个step的结束，更新参数并保存日志
                if micro_step % train_config['n_batches_per_step'] == 0:
                    step += 1
                    # 梯度裁剪保证训练稳定
                    for optimizer, _ in optimizers_with_lrscale:
                        scaler.unscale_(optimizer) # 将梯度去缩放回原始大小防止梯度裁剪错误
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # 优化器更新参数
                    for optimizer, _ in optimizers_with_lrscale:
                        scaler.step(optimizer)
                    if model_type == "NGPT":
                        model.normalize() # NGPT需要在每个训练步进行参数归一化
                    # 更新缩放系数
                    scaler.update()
                    # 保存日志
                    open(log_fname, 'a').write(f'TRAIN,{step},{lr},{total_loss},{time.time()},{grad_norm}\n')
                    # 定期进行验证并保存检查点
                    if step % train_config["validation_interval"] == 0:
                        checkpoint_name = f'checkpoint_{step}_{total_loss:.4f}'
                        model.save_pretrained(os.path.join(config_dir, checkpoint_name))
                        print(f'==> Saved checkpoint to {checkpoint_name}')


    except KeyboardInterrupt:
        print('Training interrupted.')
        # 保存未使用的数据集
        assert isinstance(train_loader.dataset, SFTDataset)
        used_indexes = train_loader.dataset.get_used_indexes()
        dataset_path = os.path.dirname(os.path.join(config_dir, train_config['dataset_path']))
        lst_name = os.path.join(dataset_path, f'train{step}_used.lst')
        with open(lst_name, 'w') as f:
            for i in tqdm(used_indexes):
                f.write(f'{i}\n')
        print(f"==> Unused indexes saved to {lst_name}")
        state_dict_full = {}
        for i, (optimizer, _) in enumerate(optimizers_with_lrscale):
            optimizer_state = optimizer.state_dict()
            state_dict_full[i] = optimizer_state
        torch.save(state_dict_full, os.path.join(config_dir, f'optimizer_{step}.pt'))
        print(f"==> Optimizer state saved to {os.path.join(config_dir, f'optimizer_{step}.pt')}")
        print("!! REMEMBER TO UPDATE THE DATASET FILE AND CONFIG FILE TO USE THE UPDATED LIST AND CHECKPOINT !!")

    finally:
        # 保存最终的检查点
        checkpoint_name = f'checkpoint_{step}.pt'
        model.save_pretrained(os.path.join(config_dir, checkpoint_name))
        print(f'==> Saved checkpoint to {checkpoint_name}')
