import os
from multiprocessing import Process, Queue
from json import loads
from . import config
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from tqdm import tqdm

def get_messages(history: list[list[str]]):
    messages = []
    for conversation in history:
        messages.append({'role': 'user', 'content': conversation[0]})
        messages.append({'role': 'assistant', 'content': conversation[1]})
    return messages

def worker(q: Queue, tokenizer: PreTrainedTokenizer, bin_path: str, mask_path: str, max_length: int, i: int):
    bin_path = bin_path + f"{i}.part"
    mask_path = mask_path + f"{i}.part"

    with open(bin_path, 'wb') as f_bin, open(mask_path, 'wb') as f_mask:
        while True:
            line = q.get()
            if line is None:
                break
            data = loads(line)
            if data["num_utter"] < 2:
                continue # 过滤掉单轮对话
            ids = []
            mask = []
            data["history"].append([data["instruction"] + data["input"], data["output"]])
            d = tokenizer.apply_chat_template(
                get_messages(data["history"]),
                return_assistant_tokens_mask=True,
                return_tensors="np",
                return_dict=True
            )
            raw_ids = d.input_ids.astype(np.uint16).squeeze()
            pad_len = max_length + 1 - raw_ids.shape[0] % (max_length + 1)
            padded_ids = np.pad(raw_ids, (0, pad_len), 'constant', constant_values=config.SPECIAL_TOKENS["<pad>"])
            padded_ids.tofile(f_bin)
            raw_mask = d.assistant_masks.astype(np.bool).squeeze()
            padded_mask = np.pad(raw_mask, (0, pad_len), 'constant', constant_values=False)
            padded_mask.tofile(f_mask)

def preprocess_zhsft(text_path: str, bin_path: str, mask_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    q: Queue[str | None] = Queue(maxsize=10000)
    processes = []
    for i in range(num_workers):
        p = Process(target=worker, args=(q, tokenizer, bin_path, mask_path, max_length, i))
        p.start()
        processes.append(p)
    with open(text_path) as f:
        for line in tqdm(f):
            q.put(line)
    for _ in range(num_workers):
        q.put(None)
    for p in processes:
        p.join()
    # 将各个部分合并并删除
    with open(bin_path, 'wb') as f_bin, open(mask_path, 'wb') as f_mask:
        for _ in range(num_workers):
            with open(bin_path + f"{_}.part", 'rb') as f_bin_part, open(mask_path + f"{_}.part", 'rb') as f_mask_part:
                f_bin.write(f_bin_part.read())
                f_mask.write(f_mask_part.read())
            os.remove(bin_path + f"{_}.part")
            os.remove(mask_path + f"{_}.part")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        # 为获取最大长度再读取配置文件过于复杂也没必要所以直接接受命令行输入
        print("Usage: python -m minilm2.utils.preprocess_zhsft <encoder_path> <text_path> <bin_path> <mask_path> <max_length>")
        exit(1)
    encoder_path = sys.argv[1]
    text_path = sys.argv[2]
    bin_path = sys.argv[3]
    mask_path = sys.argv[4]
    max_length = int(sys.argv[5])
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)
    preprocess_zhsft(text_path, bin_path, mask_path, tokenizer, max_length)
