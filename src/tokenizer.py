import os
import re
import json
from typing import Iterable, Callable, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class Tokenizer:
    """A word-level tokenizer with vocabulary management, normalization, and serialization.

    ***WordFlow: Word-Level Language Modeling with RNNs: github.com/hoom4n/wordflow***

    This tokenizer splits text on whitespace after optional lowercasing and a custom
    preprocessing step. It supports special tokens (e.g., ``<pad>``, ``<unk>``, ``<bos>``,
    ``<eos>``, ``<num>``) and can build a fixed-size vocabulary from a corpus by retaining
    the most frequent tokens.

    Attributes:
        word2idx (dict): Mapping from token string to integer ID.
        idx2word (list): List of token strings, where index is the token ID.
        max_tokens (int): Maximum vocabulary size (including special tokens).
        unk_token (str): Token representing unknown words.
        special_tokens (list[str]): List of special tokens that are always added first.
        unk_id (int): ID of the unknown token.
        preprocessor (Callable or None): Optional function applied to raw text before
            tokenization.
        lowercase (bool): Whether to lowercase text before tokenization.

    Example:
        >>> tokenizer = Tokenizer(max_tokens=10000)
        >>> tokenizer.build_vocab(["Hello world!", "Hello again."])
        >>> ids = tokenizer.encode("Hello world")
        >>> print(ids)
        [2, 3]
        >>> print(tokenizer.decode(ids))
        'hello world'
    """

    def __init__(
        self,
        max_tokens: int = 30_000,
        unk_token: str = "<unk>",
        special_tokens: list[str] = ["<pad>", "<unk>", "<bos>", "<eos>", "<num>"],
        lowercase: bool = True,
        preprocessor: Callable = None,
    ):
        """Initialize the tokenizer with configuration.

        ***WordFlow: Word-Level Language Modeling with RNNs: github.com/hoom4n/wordflow***
        
        Args:
            max_tokens: Maximum number of tokens in the vocabulary, including special tokens.
                Defaults to 30,000.
            unk_token: String used for unknown tokens. Must be present in ``special_tokens``.
                Defaults to ``'<unk>'``.
            special_tokens: List of special tokens that are added to the vocabulary first.
                Defaults to ``['<pad>', '<unk>', '<bos>', '<eos>', '<num>']``.
            lowercase: If ``True``, lowercase all text before tokenization. Defaults to ``True``.
            preprocessor: Optional callable that receives the raw text and returns processed
                text. Applied before lowercasing. If ``None``, no custom preprocessing is done.

        Raises:
            AssertionError: If ``unk_token`` is not in ``special_tokens``.
            AssertionError: If ``preprocessor`` is provided but not callable.
        """
        assert unk_token in special_tokens
        if preprocessor is not None:
            assert callable(preprocessor)

        self.word2idx = {}
        self.idx2word = []
        self.max_tokens = max_tokens
        self.unk_token = unk_token
        self.special_tokens = special_tokens
        self.preprocessor = preprocessor
        self.lowercase = lowercase
        self._add_special_tokens()
        self.unk_id = self.word2idx[self.unk_token]

    def _add_token(self, token: str) -> None:
        """Add a token to the vocabulary if it is not already present.

        Args:
            token: Token string to add.
        """
        if token not in self.word2idx:
            self.idx2word.append(token)
            self.word2idx[token] = len(self.idx2word) - 1

    def _add_special_tokens(self) -> None:
        """Add all special tokens to the vocabulary."""
        if self.special_tokens is not None:
            for token in self.special_tokens:
                self._add_token(token)

    def get_vocab(self) -> list:
        """Return the vocabulary as a list of tokens.

        Returns:
            List of token strings in the order they were added (ID = index).
        """
        return self.idx2word

    def get_vocab_size(self) -> int:
        """Return the current number of tokens in the vocabulary.

        Returns:
            Vocabulary size (int).
        """
        return len(self.idx2word)

    def normalizer(self, text: str) -> str:
        """Apply preprocessing and optional lowercasing to the input text.

        Args:
            text: Raw input string.

        Returns:
            Normalized string ready for tokenization.
        """
        if self.preprocessor:
            text = self.preprocessor(text)
        if self.lowercase:
            text = text.lower()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Normalize and split text into a list of tokens.

        Tokenization is performed by splitting on whitespace after normalization.

        Args:
            text: Raw input string.

        Returns:
            List of string tokens.

        Raises:
            AssertionError: If ``text`` is not a string.
        """
        assert isinstance(text, str)
        text = self.normalizer(text)
        return text.split()

    def _build_counter(self, iterator: Iterable[str]) -> Counter:
        """Build a token frequency counter from an iterable of texts.

        Args:
            iterator: Iterable of raw strings (e.g., sentences, documents).

        Returns:
            ``collections.Counter`` mapping tokens to their frequencies.
        """
        counter = Counter()
        for text in iterator:
            for token in self.tokenize(text):
                counter[token] += 1
        return counter

    def build_vocab(self, iterator: Iterable[str]) -> None:
        """Build the vocabulary from a corpus.

        The most frequent tokens up to ``max_tokens`` (minus the special tokens) are
        added to the vocabulary. The special tokens are always kept and were added
        during initialization.

        Args:
            iterator: Iterable of strings. Each element is a raw text (e.g., a sentence
                or a document).

        Raises:
            AssertionError: If ``iterator`` is a string (should be an iterable of strings).
            AssertionError: If any element of the iterator is not a string.

        Prints:
            A message indicating how many tokens were added to the vocabulary.
        """
        assert not isinstance(iterator, str)
        assert all(isinstance(x, str) for x in iterator)
        counter = self._build_counter(iterator)
        most_common = counter.most_common(self.max_tokens - len(self.special_tokens))
        for token, _ in most_common:
            self._add_token(token)
        logger.info(f"*** {len(self.idx2word):,} tokens added to vocab ***")

    def save(self, save_path: str) -> None:
        """Save the tokenizer configuration and vocabulary to a JSON file.

        Args:
            save_path: Path to the output JSON file.

        Prints:
            A success message with the save path or an error message on failure.
        """
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": "Word-Level Tokenizer (GiTHUB.com/HooM4N/WordFlow)",
                        "vocab_size": len(self.idx2word),
                        "special_tokens": self.special_tokens,
                        "unk_token": self.unk_token,
                        "vocab": self.idx2word,
                    },
                    f,
                )
            logger.info(f"*** tokenizer saved to {save_path} ***")
        except Exception as e:
            logger.warning(f"*** error saving tokenizer at {save_path}: {e} ***")

    @classmethod
    def load_from_file(cls, load_path: str) -> "Tokenizer":
        """Load a tokenizer from a previously saved JSON file.

        Args:
            load_path: Path to the JSON file produced by :meth:`save`.

        Returns:
            A fully initialized ``Tokenizer`` instance, or ``None`` if loading fails.

        Raises:
            AssertionError: If the file does not exist.
        """
        assert os.path.exists(load_path)
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            tokenizer = cls()
            tokenizer.idx2word = data["vocab"]
            tokenizer.word2idx = {w: i for i, w in enumerate(tokenizer.idx2word)}
            tokenizer.unk_token = data["unk_token"]
            tokenizer.unk_id = tokenizer.word2idx[tokenizer.unk_token]
            tokenizer.special_tokens = data["special_tokens"]
            return tokenizer
        except Exception as e:
            logger.warning(f"*** error loading tokenizer from {load_path}: {e} ***")
            return None

    def encode(self, text: str) -> list[int]:
        """Convert a text into a list of token IDs.

        Unknown tokens are mapped to ``unk_id``.

        Args:
            text: Raw input string.

        Returns:
            List of integer token IDs.

        Raises:
            AssertionError: If ``text`` is not a string.
        """
        assert isinstance(text, str)
        return [self.word2idx.get(t, self.unk_id) for t in self.tokenize(text)]

    def decode(self, token_ids: list[int]) -> str:
        """Convert a list of token IDs back into a space-separated string.

        IDs outside the valid range are replaced by ``unk_token``.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            A single string with tokens joined by spaces.

        Raises:
            AssertionError: If ``token_ids`` is not a list.
        """
        assert isinstance(token_ids, list)
        vocab_len = len(self.idx2word)
        return " ".join(
            self.idx2word[i] if 0 <= i < vocab_len else self.unk_token
            for i in token_ids
        )

    def token_to_id(self, token: str) -> int:
        """Return the ID for a given token, or ``unk_id`` if unknown.

        Args:
            token: Token string.

        Returns:
            Integer ID.
        """
        return self.word2idx.get(token, self.unk_id)

    def id_to_token(self, id_: int) -> str:
        """Return the token string for a given ID, or ``unk_token`` if out of range.

        Args:
            id_: Token ID.

        Returns:
            Token string.
        """
        return self.idx2word[id_] if 0 <= id_ < len(self.idx2word) else self.unk_token