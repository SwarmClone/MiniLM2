import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from minilm2.llm import config

def validate(model: torch.nn.Module, val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], batch_size: int = 2) -> float:
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    losses: list[float] = []
    with torch.no_grad(), tqdm(val_loader) as pbar:
        for x, y in pbar:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='mean')
            losses.append(loss.item())
            pbar.set_description(f"val loss: {loss.item():.4f}")
            del loss, x, y, logits
    return sum(losses) / len(losses)

if __name__ == '__main__':
    pass # TODO: 接受一个模型测试点路径，返回测试结果
