import tokenizers
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase

def hf_tokenizer_train(
    iterator,
    model: str = "WordLevel",
    vocab_size: int = 15_000, 
    special_tokens: list = ["<pad>", "<unk>", "<bos>", "<eos>", "<num>"],
):
    """
    models: WordLevel , BPE , WordPiece
    """
    trainers = {"WordLevel":"WordLevelTrainer", "BPE":"BpeTrainer", "WordPiece":"WordPieceTrainer"}
    assert model in trainers
    tokenizer = Tokenizer(getattr(tokenizers.models, model)(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([
        Lowercase()
    ])
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.train_from_iterator(
        iterator, 
        getattr(tokenizers.trainers, trainers.get(model))(
            vocab_size = vocab_size, special_tokens=special_tokens
        )
    )
    print(f"*** vocab size: {tokenizer.get_vocab_size():,} ***")
    return tokenizer