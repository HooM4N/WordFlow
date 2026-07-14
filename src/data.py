import string
import pandas as pd

_PUNK_TABLE = str.maketrans({p: f" {p} " for p in string.punctuation})

def text_preprocessor(text:str) -> str:
    text = text.lower()
    text = text.replace("\n\n"," XXXBREAKLINEXXX ")
    return text.translate(_PUNK_TABLE)


def get_data(
        data_path:str,
        
):
    df = pd.read_parquet(DATA_PATH)[:15_000]
    pass