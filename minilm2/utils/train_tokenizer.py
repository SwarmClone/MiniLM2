from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from . import config
import os

"""
def train_tokenizer(f: StringIO, extra_tokens: list[str] | None) -> Tokenizer:
    if extra_tokens is None:
        extra_tokens = []
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        special_tokens=list(config.SPECIAL_TOKENS.keys()),
        vocab_size=32768 - len(extra_tokens)
    )
    tokenizer.train_from_iterator(f, trainer=trainer)
    tokenizer.add_tokens(extra_tokens)
    return tokenizer

"""

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m minilm2.utils.train_tokenizer <text path> <tokenizer path>")
        exit(1)
    path, tokenizer_path = sys.argv[1:]
    print(f"Training tokenizer from {path}")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split(r"\s+", "merged_with_previous")
    trainer = BpeTrainer(
        vocab_size=32768,
        special_tokens=[*config.SPECIAL_TOKENS.keys()],
    )
    tokenizer.train_from_iterator(open(path), trainer=trainer)
    os.makedirs(tokenizer_path, exist_ok=True)
    open(tokenizer_path + "/tokenizer.json", "w").write(tokenizer.to_str())
    tfs_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path + "/tokenizer.json")
    tfs_tokenizer.save_pretrained(tokenizer_path)