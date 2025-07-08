from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os

def preprocess_text(text: str, out_fname: str, tokenizer: PreTrainedTokenizer):
    ids = tokenizer.encode(text)
    seq = np.array(ids, dtype=np.uint16)
    seq.tofile(out_fname)

def preprocess_openwebtext(text_path: str, bin_path: str, tokenizer: PreTrainedTokenizer):
    with (  open(text_path, 'r', encoding='utf-8') as f,
            Pool(16) as pool):
        text = ""
        fid = 0
        for line in tqdm(f):
            if len(line) > 3:
                text += line
            if len(text) > 1e6:
                pool.apply_async(preprocess_text, args=(text, f"{fid}.bin", tokenizer))
                fid += 1
                text = ""
            
        pool.close()
        pool.join()
    for i in tqdm(range(fid)):
        with (  open(f"{i}.bin", 'rb') as bin_part,
                open(bin_path, 'ab') as f_bin):
            data = bin_part.read()
            f_bin.write(data)
            os.remove(f"{i}.bin")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python -m minilm2.utils.preprocess_openwebtext <encoder_path> <text_path> <bin_path>")
        exit(1)
    encoder_path = sys.argv[1]
    text_path = sys.argv[2]
    bin_path = sys.argv[3]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)
    preprocess_openwebtext(text_path, bin_path, tokenizer)
