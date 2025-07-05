from torch.utils.data import Dataset
import torch
import json
import os
from ..config import Config
config = Config()
from transformers import PreTrainedTokenizerBase

class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
            self,
            data_path: str,
            max_length: int,
            tokenizer: PreTrainedTokenizerBase,
            epochs: int = 1):
        self.data = [*map(json.loads, open(data_path))] * epochs
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.data[index]
        instruction = "问：" + d["instruction"]
        output = d["output"]
        x = self.tokenizer.encode(instruction)[-self.max_length + 1:]
        y = self.tokenizer.encode(output)[-self.max_length + 1:]
        x = x + [config.SPECIAL_TOKENS["<eos>"]] + [config.SPECIAL_TOKENS["<pad>"]] * (self.max_length - 1 - len(x))
        y = y + [config.SPECIAL_TOKENS["<eos>"]] + [config.SPECIAL_TOKENS["<pad>"]] * (self.max_length - 1 - len(y))
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    x_list, y_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return x, y

if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer # type: ignore
    if len(sys.argv) < 3:
        print("Usage: python -m minilm2.llm.dataset_embedding <tokenizer_path> <data_path>")
        exit(1)
    tokenizer_path = sys.argv[1]
    data_path = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Loading dataset from {data_path}...")
    dataset = PairDataset(data_path, 1024, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for x, y in dataloader:
        print(x)
        print(y)
        print(x.shape, y.shape)
        for i in range(2):
            print(tokenizer.decode(x[i].tolist()))
            print(tokenizer.decode(y[i].tolist()))
        try:
            input()
        except EOFError:
            break