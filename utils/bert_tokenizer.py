from transformers import BertTokenizer
from typing import Dict, List, Tuple

class WordTokenizer:
    def __init__(self) -> None:
        """Constructs a tokenizer that simply splits each character"""
        self.tokenizer = lambda x: list(x)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        return self.tokenizer(text)

class ExpressionBertTokenizer(BertTokenizer):
    """
    Constructs a bert-based tokenizer used for the Regression Transformer.

    Args:
        vocab_file: path to a token per line vocabulary file.
    """

    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        pad_even: bool = True,
        **kwargs,
    ) -> None:
        """Constructs an ExpressionTokenizer.

        Args:
            vocab_file: vocabulary file containing tokens.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
            pad_even (bool): Boolean indicating whether sequences of odd length should
                be padded to have an even length. Neede for PLM in XLNet. Defaults to
                True.
        """
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=False,
            do_basic_tokenize=True,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.text_tokenizer = WordTokenizer()
        if pad_even:
            self.pad_even_fn = lambda x: x if len(x) % 2 == 0 else x + [self.pad_token]
        else:
            self.pad_even_fn = lambda x: x

    @property
    def vocab_list(self) -> List[str]:
        """List vocabulary tokens.

        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a text representing an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        splitted_expression = text.split(' ')
        return splitted_expression


    def add_padding_tokens(
        self, token_ids: List[int], max_length: int, padding_right: bool = True
    ) -> List[int]:
        """Adds padding tokens to return a sequence of length max_length.

        By default padding tokens are added to the right of the sequence.

        Args:
            token_ids: token indexes.
            max_length: maximum length of the sequence.
            padding_right: whether the sequence is padded on the right. Defaults to True.

        Returns:
            padded sequence of token indexes.
        """
        padding_ids = [self.pad_token_id] * (max_length - len(token_ids))
        if padding_right:
            return token_ids + padding_ids
        else:
            return padding_ids + token_ids