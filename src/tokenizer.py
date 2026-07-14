import os, re, json
from typing import Iterable, Callable, List
from collections import Counter

class Tokenizer():
    def __init__(
        self, max_tokens: int = 30_000, 
        unk_token: str = "<unk>", 
        special_tokens: list[str] = ["<pad>", "<unk>", "<bos>", "<eos>", "<num>"], 
        lowercase: bool = True, 
        preprocessor: Callable = None
    ):
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

    def _add_token(self, token:str):
        if token not in self.word2idx:
            self.idx2word.append(token)
            self.word2idx[token] = len(self.idx2word)-1

    def _add_special_tokens(self):
        if self.special_tokens is not None:
            for token in self.special_tokens:
                self._add_token(token)

    def get_vocab(self) -> list:
        return self.idx2word

    def get_vocab_size(self) -> int:
        return len(self.idx2word)

    def normalizer(self, text:str) -> str:
        if self.preprocessor:
            text = self.preprocessor(text) 
        if self.lowercase: 
            text = text.lower() 
        return text

    def tokenize(self, text:str) -> list[str]:
        assert isinstance(text, str)
        text = self.normalizer(text)
        return text.split()

    def _build_counter(self, iterator: Iterable[str]) -> Counter:
        counter = Counter()
        for text in iterator:
            for token in self.tokenize(text):
                counter[token] += 1
        return counter

    def build_vocab(self, iterator: Iterable[str]):
        assert not isinstance(iterator, str)
        assert all(isinstance(x, str) for x in iterator)
        counter = self._build_counter(iterator)
        most_common = counter.most_common(self.max_tokens - len(self.special_tokens))
        for token, _ in most_common:
            self._add_token(token)
        print(f"*** {len(self.idx2word):,} tokens added to vocab ***")

    def save(self, save_path: str):
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "name": "Word-Level Tokenizer (GiTHUB.com/HooM4N/WordFlow)",
                    "vocab_size": len(self.idx2word),
                    "special_tokens": self.special_tokens,
                    "unk_token": self.unk_token,
                    "vocab": self.idx2word,
                },f)
                print(f"*** tokenizer saved to {save_path} ***")
        except Exception as e:
            print(f"*** error saving tokenizer at {save_path}: {e} ***")
    
    @classmethod
    def load_from_file(cls, load_path: str):
        assert os.path.exists(load_path)
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            tokenizer = cls()
            tokenizer.idx2word = data["vocab"]
            tokenizer.word2idx = {w:i for i,w in enumerate(tokenizer.idx2word)}
            tokenizer.unk_token = data["unk_token"]
            tokenizer.unk_id = tokenizer.word2idx[tokenizer.unk_token]
            tokenizer.special_tokens = data["special_tokens"]
            return tokenizer
        except Exception as e:
            print(f"*** error loading tokenizer from {load_path}: {e} ***")
            return None

    def encode(self, text:str) -> list[int]:
        assert isinstance(text, str)
        return [self.word2idx.get(t, self.unk_id) for t in self.tokenize(text)]

    def decode(self, token_ids: list[int]) -> str:
        assert isinstance(token_ids, list)
        vocab_len = len(self.idx2word)
        return " ".join(self.idx2word[i] if 0 <= i < vocab_len else self.unk_token for i in token_ids)

    def token_to_id(self, token:str) -> int:
        return self.word2idx.get(token, self.unk_id)

    def id_to_token(self, id_:int) -> str:
        return self.idx2word[id_] if 0 <= id_ < len(self.idx2word) else self.unk_token