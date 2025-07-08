from transformers import ( # type: ignore
    PreTrainedTokenizerFast,
    TensorType
)
from transformers.tokenization_utils import ( # type: ignore
    PreTokenizedInput,
    EncodedInput,
    TextInput,
    PaddingStrategy,
    TruncationStrategy
)

class MiniLM2Tokenizer(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)
    
    def _decode(self, token_ids, **kwargs):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))
    
    def encode(self, text: TextInput | PreTokenizedInput | EncodedInput, text_pair: TextInput | PreTokenizedInput | EncodedInput | None = None, add_special_tokens: bool = True, padding: bool | str | PaddingStrategy = False, truncation: bool | str | TruncationStrategy | None = None, max_length: int | None = None, stride: int = 0, padding_side: str | None = None, return_tensors: str | TensorType | None = None, **kwargs) -> list[int]:
        if isinstance(text, str):
            text = text.expandtabs(4)
        elif isinstance(text, list) and isinstance(text[0], str):
            text = [t.expandtabs(4) for t in text]
        if isinstance(text_pair, str):
            text_pair = text_pair.expandtabs(4)
        elif isinstance(text_pair, list) and isinstance(text_pair[0], str):
            text_pair = [t.expandtabs(4) for t in text_pair]
        return super().encode(text, text_pair, add_special_tokens, padding, truncation, max_length, stride, padding_side, return_tensors, **kwargs)
