import re
import string
import logging

logger = logging.getLogger(__name__)

_PUNK_TABLE = str.maketrans({p: f" {p} " for p in string.punctuation})

def text_preprocessor(text: str) -> str:
    """
    A simple text preprocessing function, it lowercase, merger multiple breaklines,
    preserve newline markers as <breakline> and end-of-sequence tokens as <eos>.

    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    text = text.lower()
    text = re.sub(r"\n+", " \n ", text)
    text = text.replace("\n", " BREAKLINEARKER ").replace("<eos>", " EOSMARKER ")
    text = text.translate(_PUNK_TABLE)
    return text.replace(" EOSMARKER ", " <eos> ").replace(" BREAKLINEARKER ", " <breakline> ")

def get_data(data_path: str) -> str:
    """
    Simple data pipeline that loads the text file, process and return as string.

    Args:
        data_path (str): path to text data file

    WordFlow: Word-Level Language Modeling with RNNs GiTHub.com/HooM4N/WordFlow
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
        return text_preprocessor(corpus)
    except Exception as e:
        logger.error(f"Couldn't open data file at {data_path}: {e}")
        return ""