import re, string, unicodedata

#====================================#
#     Text Preprocessing Utility     #
#====================================#

## Punctuation translation tables ##
_norm_punks = {
        "’" : "'", '“' : '"', '”' : '"', '‘' : "'", '`' : "'",
        '—' : '-', '–' : '-'
    }
_rm_punk_table = str.maketrans({f"{p}":" " for p in string.punctuation})
_keep_punk_table = str.maketrans({f"{p}":f" {p} " for p in string.punctuation})

def _num_bucket(m):
    n = int(m.group())
    if 1500 <= n <= 2099:
        return " <year> "
    return " <num> "

## Main function ##
    
def text_preprocess(
    text: str,
    lowercase: bool = True,
    remove_parenthesized_content: bool = True,
    punctuations: str = "keep_essential",
    essential_punctuations: list = [".", "'", ","],
    collapse_multiple_dots: bool = True,
    special_artifacts_replacement: dict = {"<br /><br />" : "\n", " @,@ " : "", " @.@ " : ""},
    remove_non_ascii: bool = True,
    bucketize_numbers: bool = True,
    replace_special_tokens: bool = True,
) -> str:
    """
    ==============================================================
    == Text Preprocessing Pipeline (GitHub.com/HooM4N/WordFlow) ==
    ==============================================================
    Args:
        - text (str): 
            Input text to process.
        
        - lowercase (bool): 
            Convert all text to lowercase. Defaults to True.
        
        - remove_parenthesized_content (bool): 
            Remove content inside parentheses or brackets. Defaults to True.
        
        - punctuations (str): 
            How to handle punctuation: "keep", "remove", or "keep_essential".
        
        - essential_punctuations (list): 
            Punctuation to preserve with spacing when "keep_essential" (e.g., '.', "'", ',').
        
        - collapse_multiple_dots (bool): 
            Replace multiple dots with a single spaced dot. Defaults to True.
        
        - special_artifacts_replacement (dict): 
            Map custom strings (like HTML tags) to replacements.
        
        - remove_non_ascii (bool): 
            Remove non-ASCII characters. Defaults to True.
        
        - bucketize_numbers (bool): 
            Replace numbers with categorical tokens (e.g., "123" → "num").
        
        - replace_special_tokens (bool): 
            Replace sequences like `xxspecialxxspecialxx` with `<special>`.

    Returns:
        str: Cleaned and normalized text.
    """
    assert punctuations in ["keep", "remove", "keep_essential"]

    # Lowercasing
    if lowercase:
        text = text.lower()

    # Remove parenthesized & bracketed content
    if remove_parenthesized_content:
        text = re.sub("[\(\[].*?[\)\]]", " ", text) 

    # Normalize punctuations & replace artifacts
    text = text.translate(str.maketrans(_norm_punks))
    for old, new in special_artifacts_replacement.items():
        text = text.replace(old, new)

    # Punctuations
    if punctuations == "keep":
        text = text.translate(_keep_punk_table)
    elif punctuations == "remove":
        text = text.translate(_rm_punk_table)
    elif punctuations == "keep_essential":
        _keep_essential_table = str.maketrans({
            p: " . " for p in "!?;" if p not in essential_punctuations
        } | {
            p: f" {p} " for p in essential_punctuations
        } | {
            p: " " for p in string.punctuation if p not in "?!;" and p not in essential_punctuations
        })
        text = text.translate(_keep_essential_table)

    # Collapse multiple dots
    if collapse_multiple_dots:
        text = re.sub(r"\.{2,}", " . ", text) 

    # Remove non-ascii words
    if remove_non_ascii:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    # Bucketize numbers
    if bucketize_numbers:
        text = re.sub(r"\d+", _num_bucket, text) 

    # Replace special tokens
    if replace_special_tokens:
        text = re.sub(r'xxspecialxx([a-z]+)xx', r'<\1>', text)
    
    # Merge duplicated special tokens
    text = re.sub(r'(<(?:num|year|person|loc)>)(\s+\1)+', r'\1', text) 

    return " ".join(text.split())

#==================================#
#     SpaCy Entity Replacement     #
#==================================#
_nlp = None

def _get_nlp(model_name: str="en_core_web_md"):
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
        _nlp = spacy.load(model_name)
    return _nlp

def replace_entities(
    text: str, 
    entities_to_replace: list[str] = ["PERSON", "GPE", "NORP", "LANGUAGE", "LOC"]
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

    for ent in reversed(doc.ents):
        if ent.label_ in entities_to_replace:
            start, end = ent.start_char, ent.end_char
            new_text = new_text[:start] + f" xxspecialxx{ent.label_.lower()}xx " + new_text[end:]
    return new_text