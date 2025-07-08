import os
from multiprocessing import Process, Queue
from json import loads
from . import config
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from tqdm import tqdm

def worker(q: Queue, tokenizer: PreTrainedTokenizer, bin_path: str, mask_path: str, max_length: int, i: int):
    bin_path = bin_path + f"{i}.part"
    mask_path = mask_path + f"{i}.part"

    separator_ids = tokenizer.encode("\n" * 3)
    human_separator_mask = [0] * len(separator_ids)
    ai_separator_mask = [1] * len(separator_ids)
    human_prefix_ids = tokenizer.encode(config.HUMAN_PREFIX)
    human_prefix_mask = [0] * len(human_prefix_ids)
    ai_prefix_ids = tokenizer.encode(config.AI_PREFIX)
    ai_prefix_mask = [0] * len(ai_prefix_ids)

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
            for i in range(len(data["history"])):
                # 人类输入
                human_input_codes = tokenizer.encode(data["history"][i][0])
                ids += human_prefix_ids + human_input_codes + separator_ids
                mask += human_prefix_mask +  [0] * len(human_input_codes) + human_separator_mask
                # AI输出
                ai_input_codes = tokenizer.encode(data["history"][i][1])
                ids += ai_prefix_ids + ai_input_codes + separator_ids
                mask += ai_prefix_mask + [1] * (len(ai_input_codes)) + ai_separator_mask
            raw_ids = np.array(ids, dtype=np.uint16)
            pad_len = max_length + 1 - raw_ids.shape[0] % (max_length + 1)
            padded_ids = np.concat((raw_ids, np.zeros(pad_len, dtype=np.uint16)))
            padded_ids.tofile(f_bin)
            raw_mask = np.array(mask, dtype=np.bool)
            padded_mask = np.concat((raw_mask, np.zeros(pad_len, dtype=np.bool)))
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
