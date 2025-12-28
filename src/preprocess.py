import re
import unicodedata

#===============================#
#     Text Cleaning Utility     #
#===============================#

def text_cleaner(text: str) -> str:
    """
    ==========================================================================
    == Text Cleaning & Normalization Utility (GitHub.com/HooM4N/CausalLSTM) ==
    ==========================================================================
    Functions:
        - Lowercase, normalize non-ascii words
        - Remove parenthesized & bracketed content, normalize punctuations
        - Expand contractions, keep essential punctuations (dot, comma, apastrophese)
        - Seprate punks from words, bucketize numbers
        - Preserve sepecial tokens with format xxspecialxx<token>xx
    """
    text = text.lower()
    
    text = re.sub("[\(\[].*?[\)\]]", " ", text) # remove parenthesized & bracketed content
    
    norm_punks = {
        "!" : ".", "?" : ".", ";" : " . ", "’" : "'",
        '“' : '"', '”' : '"', '‘' : "'", '`' : "'",
        '—' : '-', '–' : '-'
    }
    text = text.translate(str.maketrans(norm_punks)) # normalize punks
    
    text = re.sub(r"\.{2,}", " . ", text) # collapse multiple dots
    
    replacements = {
        "n't" : " not ", "'s" : " 's ", "'re" : " are ", "'d" : " 'd ",
        "no." : " number", "'ll" : " will ", "'ve" : " have ", "'m" : " am ",
        "u.s." : " america ", "--" : " , ", "cannot" : " can not",
        "<br /><br />" : "\n", " @,@ " : "", " @.@ " : "",
        "mr." : " xxspecialxxtitlexx ", "mrs." : " xxspecialxxtitlexx ",
        "ms." : " xxspecialxxtitlexx ", "dr." : " xxspecialxxtitlexx ", 
        "prof." : " xxspecialxxtitlexx ", "ph.d." : " xxspecialxxtitlexx ",
        "m.d." : " xxspecialxxtitlexx "
    }
     
    for old, new in replacements.items(): # expand contractions
        text = text.replace(old, new)

    text = text.translate(str.maketrans({ch: " " for ch in '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'})) # remove punks except essentials
    
    text = text.replace(".", " . ").replace(",", " , ") # seprated dots & commas
    
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii") # normalize non ascii chars
    
    def num_bucket(m):
        n = int(m.group())
        if 1500 <= n <= 2099:
            return " <year> "
        return " <num> "
        
    text = re.sub(r"\d+", num_bucket, text) # bucketize numbers
    text = re.sub(r'xxspecialxx([a-z]+)xx', r'<\1>', text) # replace entities with special tokens
    text = re.sub(r'(<(?:num|year|person|loc)>)(\s+\1)+', r'\1', text) # merge duplicated special tokens

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