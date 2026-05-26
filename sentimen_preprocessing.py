"""
Modul pra-pemrosesan teks bersama untuk analisis sentimen komentar TikTok.

Pipeline ini disengaja identik dengan notebook penelitian agar tokenizer dan
model LSTM menerima representasi teks yang sama persis, baik saat pelatihan
ulang maupun saat inferensi pada web interface.

Urutan tahap:
    clean_text -> normalize (kamus slang) -> tokenize -> remove_stopwords
    -> stemming (Sastrawi) -> lemmatization (token Inggris) -> text_final
"""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

BASE_DIR = Path(__file__).resolve().parent
SLANG_FILE = BASE_DIR / "combined_slang_words.txt"

# Kata negasi yang sengaja dipertahankan karena dapat membalik polaritas opini.
NEGATION_WORDS = {
    "tidak", "tak", "bukan", "jangan", "belum",
    "ga", "gak", "nggak", "enggak", "gk", "ngga", "tdk",
}

_ADDITIONAL_STOPWORDS = {
    "nih", "sih", "deh", "dong", "ya", "yaa", "aja", "lah", "kok", "kan", "nya",
}

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def _ensure_nltk():
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def get_stopword_set():
    _ensure_nltk()
    indonesian = set(stopwords.words("indonesian"))
    return (indonesian | _ADDITIONAL_STOPWORDS) - NEGATION_WORDS


@lru_cache(maxsize=1)
def get_stemmer():
    return StemmerFactory().create_stemmer()


@lru_cache(maxsize=1)
def get_lemmatizer():
    _ensure_nltk()
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def load_slang_dictionary(path: str = None) -> dict:
    """Muat kamus normalisasi slang.

    Berkas ``combined_slang_words.txt`` sebenarnya berformat JSON
    ``{"slang": "baku", ...}``. Versi notebook lama keliru mem-parsing-nya
    sebagai teks berdelimiter sehingga normalisasi nyaris tidak aktif. Di sini
    JSON di-parse dengan benar; ada fallback bila formatnya berubah.
    """
    slang_path = Path(path) if path else SLANG_FILE
    raw = slang_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        return {str(k).lower().strip(): str(v).lower().strip() for k, v in data.items()}
    except json.JSONDecodeError:
        mapping = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("\t", ",", ";", ":"):
                if sep in line:
                    parts = line.split(sep)
                    if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                        mapping[parts[0].strip().lower()] = parts[1].strip().lower()
                    break
        return mapping


def clean_text(text: str) -> str:
    """Lowercase, hapus URL/mention/emoji/karakter non-alfabet, rapikan spasi."""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = _EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str, slang_mapping: dict = None) -> str:
    if slang_mapping is None:
        slang_mapping = load_slang_dictionary()
    tokens = text.split()
    return " ".join(slang_mapping.get(tok, tok) for tok in tokens)


def remove_stopwords(tokens):
    sw = get_stopword_set()
    return [t for t in tokens if (t not in sw) or (t in NEGATION_WORDS)]


def stem_tokens(tokens):
    if not tokens:
        return []
    return get_stemmer().stem(" ".join(tokens)).split()


def lemmatize_tokens(tokens):
    lem = get_lemmatizer()
    out = []
    for tok in tokens:
        out.append(lem.lemmatize(tok) if re.fullmatch(r"[a-zA-Z]+", tok) else tok)
    return out


def preprocess(text: str, slang_mapping: dict = None) -> str:
    """Jalankan seluruh pipeline dan kembalikan ``text_final`` siap-model."""
    cleaned = clean_text(text)
    normalized = normalize_text(cleaned, slang_mapping)
    tokens = normalized.split()
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    tokens = [t for t in tokens if t.strip()]
    return " ".join(tokens).strip()


if __name__ == "__main__":
    samples = [
        "Sri Mulyani guru beban negara, di bayar berapa min?",
        "kasian bu sri di fitnah, ibu terbaik sehat selalu",
        "ini AI kok, jangan mudah percaya",
    ]
    slang = load_slang_dictionary()
    print(f"Entri kamus slang: {len(slang)}")
    for s in samples:
        print(f"  {s!r}\n  -> {preprocess(s, slang)!r}")
