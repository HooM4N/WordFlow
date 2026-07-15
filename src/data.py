import re
import string
import logging

logger = logging.getLogger(__name__)

_PUNK_TABLE = str.maketrans({p: f" {p} " for p in string.punctuation})

def text_preprocessor(text: str) -> str:
    """
    Preprocesses raw text for WordFlow tokenization.

    Args:
        text (str): The raw input text corpus.
        
    Returns:
        str: Preprocessed and normalized text ready for tokenization.
        
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    text = text.lower()
    text = re.sub(r"\n+", " \n ", text)
    text = text.replace("\n", " BREAKLINE ").replace("<eos>", " EOSMARKER ")
    text = text.translate(_PUNK_TABLE)
    return text.replace(" EOSMARKER ", " <eos> ")

def get_data(data_path: str) -> str:
    """
    Loads and preprocesses text data from a given file path.
    
    Args:
        data_path (str): Path to the raw text file (e.g., tiny_stories.txt).
        
    Returns:
        str: Preprocessed text corpus as a single continuous string.
        
    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
        return text_preprocessor(corpus)
    except Exception as e:
        logger.error(f"Couldn't open data file at {data_path}: {e}")
        return ""