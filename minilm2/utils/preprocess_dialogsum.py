import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from ..config import Config
config = Config()
import json

def get_assistant_message(summaries: list[dict[str, str]]) -> str:
    assistant_message = ""
    for summary in summaries:
        assistant_message += f"{summary['topic']}：{summary['summary']}\n"
    return assistant_message[:-1]

prompt_template = """请总结以下对话内容，一行一点：
{dialogue}"""

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print('Usage: python -m minilm2.utils.preprocess_dialogsum <text_path> <tokenizer_path> [max_length=1024]')
        sys.exit(1)
    _, text_path, tokenizer_path, *others = sys.argv
    if len(others) == 1:
        max_length = int(others[0])
    else:
        max_length = 1024
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    with open(text_path) as f_in, open(text_path + '.bin', 'wb') as f_out, open(text_path + '.mask', 'wb') as f_mask:
        for line in tqdm(f_in):
            data = json.loads(line)
            summaries: list[dict[str, str]] = []
            if 'summary' in data: # 只有单条总结
                summaries = [{'topic': data['topic'], 'summary': data['summary']}]
            else:
                i = 1
                while f'summary{i}' in data:
                    summaries.append({'topic': data[f'topic{i}'], 'summary': data[f'summary{i}']})
                    i += 1
            messages = [
                {'role': 'user', 'content': prompt_template.format(dialogue=data['dialogue'])},
                {'role': 'assistant', 'content': get_assistant_message(summaries)}
            ]
            d = tokenizer.apply_chat_template(messages, return_assistant_tokens_mask=True, return_dict=True)
            ids, masks = d["input_ids"], d["assistant_masks"]
            if len(ids) > max_length + 1:
                ids = ids[len(ids) - max_length - 1:]
                masks = masks[len(masks) - max_length - 1:]
            ids += [config.SPECIAL_TOKENS["<pad>"]] * (max_length + 1 - len(ids))
            masks += [0] * (max_length + 1 - len(masks))
            print(len(ids), len(masks))
            np.array(ids, dtype=np.uint16).tofile(f_out)
            np.array(masks, dtype=np.bool).tofile(f_mask)
