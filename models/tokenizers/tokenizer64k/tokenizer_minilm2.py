from transformers import ( # type: ignore
    PreTrainedTokenizerFast,
    AutoTokenizer
)

class MiniLM2Tokenizer(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)
    
    def _decode(self, token_ids, **kwargs):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))
