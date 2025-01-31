import os, multiprocessing
from . import config
from tokenizers import Tokenizer # type: ignore
import numpy as np
from tqdm import tqdm

q: multiprocessing.Queue = multiprocessing.Queue(1000)
def worker(text_path: str, tokenizer: Tokenizer, id: int, q: multiprocessing.Queue):
    with open(text_path + f"_{id}.bin", 'wb') as f_bin:
        while True:
            t = q.get()
            if t is None:
                break
            ids = tokenizer.encode(t).ids + [config.SPECIAL_TOKENS["<eos>"]]
            np.array(ids, dtype=np.uint16).tofile(f_bin)

def preprocess_openwebtext(text_path: str, bin_path: str, tokenizer: Tokenizer):
    n_workers = 16
    workers = []
    for i in range(n_workers):
        p = multiprocessing.Process(target=worker, args=(text_path, tokenizer, i, q))
        p.start()
        workers.append(p)
    with open(text_path, 'r', encoding='utf-8') as f:
        n_blank_lines = 0
        text = ""
        for line in tqdm(f):
            line = line.strip()
            if line:
                text += line + " "
                continue
            if not text:
                continue
            n_blank_lines += 1
            if n_blank_lines >= 3:
                q.put(text)
                text = ""
        q.put(text)
    for i in range(n_workers):
        q.put(None)
    with open(bin_path, 'wb') as f_bin:
        for i, p in enumerate(workers):
            p.join()
            with open(text_path + f"_{i}.bin", 'rb') as fi_bin:
                f_bin.write(fi_bin.read())
            os.remove(text_path + f"_{i}.bin")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python -m minilm2.utils.preprocess_openwebtext <encoder_path> <text_path> <bin_path>")
        exit(1)
    encoder_path = sys.argv[1]
    text_path = sys.argv[2]
    bin_path = sys.argv[3]
    tokenizer = Tokenizer.from_file(encoder_path)
    preprocess_openwebtext(text_path, bin_path, tokenizer)
