from transformers import AutoTokenizer, PreTrainedTokenizerBase

def test_tokenizer(tokenizer: PreTrainedTokenizerBase):
    def _wrap(text: str):
        ids = tokenizer.encode(text)
        print(ids)
        color = 0
        for i in ids:
            color += 1
            print(f"\033[0;3{color}m{tokenizer._convert_id_to_token(i)}\033[0m", end="")
            if color == 5:
                color = 0
        print(f"\n{(tokenized_len := len(ids))} / {(original_len := len(text))} = {tokenized_len/original_len:.2f}")
    return _wrap

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m minilm2.utils.test_tokenizer <path_to_tokenizer>")
        exit(1)
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
    print_tokens = test_tokenizer(tokenizer)
    if len(sys.argv) > 2:
        for text in sys.argv[2:]:
            print_tokens(text)
        exit(0)
    try:
        while True:
            text = input("--> ")
            print_tokens(text)
    except KeyboardInterrupt:
        pass
    except EOFError:
        pass
