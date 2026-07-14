import re
import string
import logging

logger = logging.getLogger(__name__)

_PUNK_TABLE = str.maketrans({p: f" {p} " for p in string.punctuation})

def text_preprocessor(text:str) -> str:
    text = text.lower()
    text = re.sub(r"\n+", " \n ", text)
    text = text.replace("\n"," BREAKLINE ").replace("<eos>"," EOSMARKER ")
    text = text.translate(_PUNK_TABLE)
    return text.replace(" EOSMARKER ", " <eos> ")

def get_data(
        data_path:str,   
) -> str:
    try:
        with open(data_path) as f:
            corpus = f.read()
        return text_preprocessor(corpus)
    except Exception as e:
        logger.error(f"couldn't open data file: {e}")
        return None