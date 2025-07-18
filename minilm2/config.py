from dataclasses import dataclass, field

@dataclass
class Config:
    SPECIAL_TOKENS: dict[str, int] = field(default_factory=lambda: {"<pad>": 0, "<eos>": 1, "<unk>": 2, "<msg>": 3, "<eom>": 4})
    NUM_WORKERS: int = 0
    DEVICE: str = "cuda"

    ENABLE_KVCACHE: bool = True

    HUMAN_PREFIX: str = "人类："
    AI_PREFIX: str = "AI："