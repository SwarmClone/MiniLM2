import math
import warnings
import time
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from flash_muon import Muon
from minilm2.llm.modeling_ngpt import NGPTConfig
from minilm2.llm.dataset import PreTrainDataset, collate_fn, from_file
from minilm2.llm.validate import validate
from minilm2.llm import config
from minilm2.llm.lr_schedule import get_lr_schedule

if __name__ == '__main__':
    import sys
    import os
    import json
    
    if len(sys.argv) < 2:
        print('Usage: python -m minilm2.llm.train <config_path>')
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
    model = (AutoModelForCausalLM.from_pretrained(os.path.join(config_dir, train_config['model_path']))
                if train_config['model_path'] else
                AutoModelForCausalLM.from_config(model_config))
    # 统计参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"==> Number of parameters: {params / 1e6:.2f}M")

    # 将模型移动到显存并编译以加速训练
    model.to(config.DEVICE)
    scaler = GradScaler(enabled=train_config['bfloat16']) # 如果启用bfloat16则启用混合精度训练
    print("==> Compiling model...")
    model.compile()
    model.train()

    # 加载数据集
    print("Loading dataset...")
    train_dataset, val_dataset = from_file(
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
        optimizers_with_lrscale.append((
            Muon(
                params=muon_params_dict.values(),
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
            for x, y in pbar:
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
                with autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=train_config["bfloat16"] # 启用bfloat16则使用混合精度训练
                    ):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        reduction='mean',
                        ignore_index=config.SPECIAL_TOKENS["<pad>"]
                    ) / train_config['n_batches_per_step']
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
                        print('==> Validating...')
                        model.eval() # 切换到验证模式
                        val_loss = validate(model, val_dataset, train_config["val_batch_size"])
                        model.train() # 在验证时切换到了推理模式，切换回来
                        print(f'==> Validation loss: {val_loss:.4f}')
                        checkpoint_name = f'checkpoint_{step}_{val_loss:.4f}.pt'
                        model.save_pretrained(os.path.join(config_dir, checkpoint_name))
                        print(f'==> Saved checkpoint to {checkpoint_name}')
                        open(log_fname, 'a').write(f'VAL,{step},{lr},{val_loss},{time.time()}\n')

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
