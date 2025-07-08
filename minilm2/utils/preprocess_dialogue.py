from . import config
from transformers import PreTrainedTokenizer, AutoTokenizer
import numpy as np
from tqdm import tqdm

def preprocess_dialogue(text_path: str, bin_path: str, mask_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
    history: list[list[str]] = []

    separator_ids = tokenizer.encode("\n" * 3)
    human_separator_mask = [0] * len(separator_ids)
    ai_separator_mask = [1] * len(separator_ids)
    human_prefix_ids = tokenizer.encode(config.HUMAN_PREFIX)
    human_prefix_mask = [0] * len(human_prefix_ids)
    ai_prefix_ids = tokenizer.encode(config.AI_PREFIX)
    ai_prefix_mask = [0] * len(ai_prefix_ids)

    with (  open(text_path, 'r', encoding='utf-8') as f,
            open(bin_path, 'wb') as f_bin,
            open(mask_path, 'wb') as f_mask):
            for line in tqdm(f):
                line = line.strip().replace("：", ": ")
                if set(line[3:]) == {"嗯", "。"}:
                    continue # 过滤无意义对话
                if line.startswith('A: '):
                    if not history:
                        history.append([line[3:], ""])
                    else:
                        if not history[-1][1]:
                            history[-1][0] += line[3:]
                        else:
                            history.append([line[3:], ""])
                elif line.startswith('B: '):
                    if not history:
                        raise ValueError("B: line without A: line")
                    else:
                        history[-1][1] += line[3:]
                else:
                    raise ValueError("Invalid line: " + line)

            ids: list[int] = []
            mask: list[int] = []
            for i, (human_utt, ai_utt) in enumerate(history):
                if not human_utt:
                    continue
                if not ai_utt:
                    continue
                human_ids = tokenizer.encode(human_utt)
                ai_ids = tokenizer.encode(ai_utt)
                human_mask = [0] * len(human_ids)
                ai_mask = [1] * len(ai_ids)
                ids += (
                    human_prefix_ids +
                    human_ids +
                    separator_ids +
                    ai_prefix_ids +
                    ai_ids +
                    separator_ids
                )
                mask += (
                    human_prefix_mask +
                    human_mask +
                    human_separator_mask +
                    ai_prefix_mask +
                    ai_mask +
                    ai_separator_mask
                )
            padding_len = max_length + 1 - len(ids) % (max_length + 1)
            f_bin.write(np.array(ids + [config.SPECIAL_TOKENS["<pad>"]] * padding_len, dtype=np.uint16).tobytes())
            f_mask.write(np.array(mask + [0] * padding_len, dtype=np.bool).tobytes())
                

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 6:
        print('Usage: python -m minilm2.utils.preprocess_dialogue <encoder_path> <text_path> <bin_path> <mask_path> <max_length>')
        exit(1)
    encoder_path = sys.argv[1]
    text_path = sys.argv[2]
    bin_path = sys.argv[3]
    mask_path = sys.argv[4]
    max_length = int(sys.argv[5])
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    preprocess_dialogue(text_path, bin_path, mask_path, tokenizer, max_length)
