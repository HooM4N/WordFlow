import re, unicodedata
   
def is_valid_row(row, len_thresh=25):
    """
    =========================================================================
    == Masks valid rows in WikiText dataset (GitHub.com/HooM4N/CausalLSTM) ==
    =========================================================================
    """
    return len(row["text"].split()) >= len_thresh

def text_cleaner(text: str) -> str:
    """
    ==========================================================================
    == Text cleaning & normalization utility (GitHub.com/HooM4N/CausalLSTM) ==
    ==========================================================================
    Functions:
        - Lowercase, normalize non ascii words
        - Remove parenthesized & bracketed content, normalize punctuations
        - Expand contractions, keep essential punctuations (dot, comma, apastrophese)
        - Seprate punks from words, bucketize numbers
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii") # normalize non ascii chars
    text = re.sub("[\(\[].*?[\)\]]", " ", text) # remove parenthesized & bracketed content
    
    norm_punks = {"!": ".", "?": ".", ";": " . ", "’":" ' ", '“': '"', '”': '"', '‘': "'", '’': "'", '`': "'", '—': '-', '–': '-'}
    text = text.translate(str.maketrans(norm_punks)) # normalize punks
    
    text = re.sub(r"\.{2,}", " . ", text) # collapse multiple dots
    
    replacements = {
        "n't": " not ", "'s": " 's ", "'re": " are ", "'d": " 'd ", "no.": " number",
        "'ll": " will ", "'ve": " have ", "'m": " am ", "u.s.": " america ",
        "--": " , ", "mr.": " ", "cannot": " can not", "<br /><br />": " ",
    }
    for old, new in replacements.items(): # expand contractions
        text = text.replace(old, new)

    text = text.translate(str.maketrans({ch: " " for ch in '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'})) # remove punks except essentials
    
    text = text.replace(".", " . ").replace(",", " , ") # seprated dots & commas

    def num_bucket(m):
        n = int(m.group())
        if 0 <= n <= 9:
            return " <1d_num> "
        if 10 <= n <= 99:
            return " <2d_num> "
        if 100 <= n <= 999:
            return " <3d_num> "           
        if 1500 <= n <= 2099:
            return " <year> "
        return " <num> "
        
    text = re.sub(r"\d+", num_bucket, text) # normalize numbers

    return " ".join(text.split())
