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

##############################
## SpaCy Entity Replacement ##
##############################
_nlp = None

def _get_nlp(model="en_core_web_md"):
    """
    ==========================================
    == Lazy loader for spaCy and nlp module ==
    ==========================================
    - Download pretrained model first:
        - python -m spacy download en_core_web_md
        - https://github.com/explosion/spacy-models/releases/
    """
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(model)
    return _nlp

def replace_entities(
    text: str,
    labels: list = ["PERSON","NORP","LANGUAGE","GPE","DATE","ORG"],
) -> str:
    """
    ==============================================================
    == Entity Replacement Utility via spaCy (GitHUB.com/HooM4N) ==
    ==============================================================
    - Download pretrained model first (using en_core_web_md)
    - Available Entities:
        * PERSON      -> people, including fictional
        * NORP        -> nationalities, religions, groups
        * FAC         -> buildings, airports, highways
        * ORG         -> companies, institutions, organizations
        * GPE         -> countries, cities, states
        * LOC         -> non-GPE locations, mountains, rivers
        * PRODUCT     -> products, objects, vehicles
        * EVENT       -> named events, wars, disasters
        * WORK_OF_ART -> titles of books, songs, movies
        * LAW         -> named legal documents
        * LANGUAGE    -> named languages
        * DATE        -> absolute/relative dates
        * TIME        -> times smaller than a day
        * PERCENT     -> percentage values
        * MONEY       -> monetary values
        * QUANTITY    -> measurements, weights, distances
        * ORDINAL     -> first, second, third, etc.
        * CARDINAL    -> numbers not otherwise categorized
    """
    nlp = _get_nlp()
    doc = nlp(text)
    new_text = text
    for ent in doc.ents:
        if ent.label_ in labels:
            new_text = new_text.replace(ent.text, f" <{ent.label_}> ")
    return new_text
