import re
import string
import logging

logger = logging.getLogger(__name__)

_PUNK_TABLE = str.maketrans({p: f" {p} " for p in string.punctuation})

# def text_preprocessor(text:str) -> str:
#     text = text.lower()
#     text = text.replace("\n\n"," XXBRKLNXX ").replace("<eos>"," XXEOSXX ")
#     return text.translate(_PUNK_TABLE)

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
    

def text_preprocessor(text: str) -> str:
    text = text.lower()
    text = text.replace("<eos>", " XXEOSXX ")

    # merge repetitive newline characters
    text = re.sub(r"\n+", " \n ", text)
    text = text.translate(_PUNK_TABLE)

    return text.replace("XXEOSXX", " <eos> ")