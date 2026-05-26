"""
Pelabelan sentimen berbasis leksikon + aturan untuk komentar TikTok isu
"Sri Mulyani / guru beban negara".

Latar belakang revisi
---------------------
Pelabelan lama memakai TextBlob atas teks hasil terjemahan Google. Banyak
komentar Bahasa Indonesia diterjemahkan menjadi polaritas 0,0 sehingga
otomatis berlabel ``neutral`` -> kelas netral membludak dan mendominasi model.

Pendekatan di sini menilai langsung kata/frasa Bahasa Indonesia (termasuk
slang, umpatan, dan idiom khas isu ini) sehingga komentar yang sebenarnya
beropini tidak lagi jatuh ke netral. Netral hanya untuk komentar yang memang
faktual/penyebutan nama/ambigu. Leksikon disusun dari pembacaan langsung
korpus + kosakata sentimen umum Bahasa Indonesia, lalu dipakai sebagai
basis pelabelan yang dapat ditinjau dan dikoreksi manual (kolom skor & alasan
disertakan pada berkas review).

API utama: ``label_comment(raw_text) -> (label, score, detail)``.
"""

import re

from sentimen_preprocessing import (
    NEGATION_WORDS,
    clean_text,
    load_slang_dictionary,
    normalize_text,
)

# --------------------------------------------------------------------------
# Leksikon kata (dinilai pada teks ter-normalisasi slang, sebelum stemming).
# Bobot 2 = sinyal kuat, 1 = sinyal sedang. Sertakan bentuk baku (hasil
# normalisasi slang) maupun bentuk mentah agar cakupan tetap luas.
# --------------------------------------------------------------------------
NEG_WORDS = {
    # umpatan / hinaan keras
    "anjing": 2, "asu": 2, "bangsat": 2, "bajingan": 2, "kontol": 2, "kntl": 2,
    "memek": 2, "ngentot": 2, "jancuk": 2, "jancok": 2, "babi": 2, "monyet": 2,
    "kunyuk": 2, "tai": 2, "taik": 2, "tahi": 2, "kotoran": 2, "keparat": 2,
    "bedebah": 2, "biadab": 2, "bejat": 2, "setan": 2, "iblis": 2, "dajjal": 2,
    "hama": 2, "tikus": 2, "maling": 2, "garong": 2, "perampok": 2, "koruptor": 2,
    "korupsi": 2, "korup": 2, "pantek": 2, "kampret": 2, "brengsek": 2,
    "sialan": 2, "busuk": 2, "tolol": 2, "goblok": 2, "goblog": 2, "gblk": 2,
    "bodoh": 2, "bego": 2, "dungu": 2, "idiot": 2, "pekok": 2, "koplak": 2,
    "geblek": 2, "dongo": 2, "pekoq": 2, "paok": 2, "bangke": 2, "pukimak": 2,
    # marah / tuduhan / sikap negatif (sedang)
    "beban": 1, "pecat": 1, "bubarkan": 1, "berkilah": 1, "ngeles": 1,
    "nyocot": 1, "cocot": 1, "lambe": 1, "jahat": 1, "kejam": 1, "fitnah": 1,
    "bohong": 1, "boong": 1, "dusta": 1, "palsu": 1, "nipu": 1, "tipu": 1,
    "ketipu": 1, "hasut": 1, "dihasut": 1, "provokator": 1, "sombong": 1,
    "songong": 1, "rakus": 1, "serakah": 1, "penjajah": 1, "penjahat": 1,
    "antek": 1, "buzzer": 1, "buzer": 1, "sogok": 1, "disogok": 1, "suap": 1,
    "disuap": 1, "penjilat": 1, "munafik": 1, "munapik": 1, "zalim": 1,
    "dzolim": 1, "najis": 1, "jijik": 1, "muak": 1, "benci": 1, "kecewa": 1,
    "geram": 1, "sengsara": 1, "tertindas": 1, "resah": 1, "gaduh": 1,
    "rusuh": 1, "ricuh": 1, "berantakan": 1, "kacau": 1, "dibodohi": 1,
    "pembodohan": 1, "ditipu": 1, "ngaco": 1, "ngawur": 1, "sembarangan": 1,
    "lancang": 1, "preet": 1, "pret": 1, "plintir": 1, "dipelintir": 1,
    "pelintir": 1, "ngebacot": 1, "menyakitkan": 1, "menyengsarakan": 1,
    # sisipan umpatan singkat (sering tidak ternormalisasi)
    "anj": 2, "anjg": 2, "anjay": 2, "anjir": 2, "anjirr": 2, "njir": 2,
    "njing": 2, "asw": 2, "asu": 2, "cok": 2, "cuk": 2, "kntl": 2, "knthl": 2,
    "jir": 1, "bacot": 1, "nyolot": 1, "kont": 2,
    # bahaya / kerusakan / nada cemas-negatif
    "bahaya": 1, "berbahaya": 1, "merusak": 1, "rusak": 1, "ngeri": 1,
    "serem": 1, "seram": 1, "mengerikan": 1, "ancur": 1, "amburadul": 1,
    "kebohongan": 1, "memfitnah": 1, "difitnah": 1, "menghina": 1, "hina": 1,
    "melecehkan": 1, "merendahkan": 1, "ngehina": 1, "menyinggung": 1,
    "tersinggung": 1, "disinggung": 1, "cair": 1, "cairr": 1, "cairrr": 1,
}

POS_WORDS = {
    # pujian / dukungan kuat
    "terbaik": 2, "cerdas": 2, "jenius": 2, "brilian": 2, "mutiara": 2,
    "srikandi": 2, "pahlawan": 2, "berjasa": 2, "mulia": 2, "bijaksana": 2,
    "kompeten": 2, "hebat": 2, "panutan": 2, "teladan": 2,
    # sikap positif / simpati (sedang)
    "pintar": 1, "pinter": 1, "pandai": 1, "bagus": 1, "baik": 1, "jujur": 1,
    "amanah": 1, "tabah": 1, "sabar": 1, "semangat": 1, "sehat": 1,
    "terimakasih": 1, "makasih": 1, "cinta": 1, "sayang": 1, "love": 1,
    "kangen": 1, "rindu": 1, "kasian": 1, "kasihan": 1, "bela": 1, "membela": 1,
    "dukung": 1, "mendukung": 1, "salut": 1, "bangga": 1, "keren": 1,
    "mantap": 1, "hormat": 1, "menghargai": 1, "hargai": 1, "ikhlas": 1,
    "berkah": 1, "sukses": 1, "sejahtera": 1, "damai": 1, "doakan": 1,
    "semoga": 1, "moga": 1, "aamiin": 1, "amin": 1, "husnul": 1, "syukur": 1,
    "bersyukur": 1, "idola": 1, "favorit": 1, "peduli": 1, "menyala": 1,
}

# Frasa multi-kata (dicocokkan pada string ter-normalisasi). Lebih spesifik
# sehingga bobotnya lebih tegas.
NEG_PHRASES = {
    "beban negara": 2, "beban rakyat": 2, "beban masyarakat": 2,
    "beban dunia": 2, "bayar berapa": 2, "berapa bayar": 2, "cair 150": 2,
    "150 juta": 2, "150 jt": 2, "100 juta": 1, "250 juta": 2, "di sogok": 2,
    "di suap": 2, "adu domba": 2,
    "pecah belah": 2, "banyak bicara": 2, "banyak omong": 2, "banyak bacot": 2,
    "tutup komen": 2, "cuci tangan": 2, "jaga mulut": 1, "asal ngomong": 2,
    "asal bicara": 2, "putar balik": 2, "balik fakta": 2, "muter balik": 2,
    "memutar balik": 2, "balik kata": 1, "putar balik kata": 2,
    "omong kosong": 2, "makan uang rakyat": 2, "makan duit rakyat": 2,
    "uang haram": 2, "duit haram": 2, "mulut mu": 1, "harimau mu": 1,
    "tidak tahu terima kasih": 2, "lupa kulit": 2, "kacang lupa": 2,
    # nada dismissif terhadap klarifikasi ("sama saja, tetap beban negara")
    "sama saja": 1, "intinya sama": 1, "maknanya sama": 1, "artinya sama": 1,
    "tetep sama": 1, "tetap sama": 1, "beda pengucapan": 1, "cuma beda": 1,
    "beda kalimat": 1, "kata halus": 1, "bahasa halus": 1, "lebih halus": 1,
    # serangan retoris "tanpa guru kamu bukan apa-apa" + varian tuduhan bayaran
    "tanpa guru": 1, "berapa juta": 2, "dapat berapa": 2, "cair berapa": 2,
}
POS_PHRASES = {
    "guru pahlawan": 2, "pahlawan tanpa tanda jasa": 2, "tanpa tanda jasa": 2,
    "sehat selalu": 2, "sehat sehat": 2, "semoga sehat": 2, "sabar bu": 2,
    "sabar ya": 1, "terima kasih": 2, "love you": 2, "ibu terbaik": 2,
    "menteri terbaik": 2, "menkeu terbaik": 2, "tetap semangat": 2,
    "semangat bu": 2, "orang baik": 1, "ibu baik": 2, "ibu cerdas": 2,
    "ibu pintar": 2, "guru berjasa": 2, "tanggung jawab negara": 1,
    "menghargai guru": 2, "hargai guru": 2, "jasa guru": 1,
    "naikin gaji guru": 2, "naikkan gaji guru": 2, "naikan gaji guru": 2,
    "guru mulia": 2, "guru hebat": 2, "pekerjaan mulia": 2,
}

# Token positif tambahan (digabung ke POS_WORDS di bawah).
POS_WORDS.update({"mencerdaskan": 1, "mendidik": 1, "tercerdas": 1})

# Sentimen emoji (sinyal sekunder, bobot 0.5). Emoji tawa sengaja dianggap
# netral karena ambigu (bisa mengejek, bisa bercanda).
POS_EMOJI = set("❤🧡💛💚💙💜🤍🖤🥰😍😘😗😚😙😊☺🙂🤗🥹🥺🙏💕💗💖💘💞💓💝💪🫶👍✨🌹🌷🌻💐👏🤩😇")
NEG_EMOJI = set("😡🤬💢👎🖕💩🤮🤢😤😠")
# Emoji tawa: pada korpus ini dominan dipakai untuk mengejek SM/pemerintah,
# sehingga diberi sinyal negatif lemah (tidak sekuat umpatan).
MOCK_EMOJI = set("🤣😂😆😹😅😏😒🙄😬😁🗿😆😄")

# Penyeragaman varian tulisan agar frasa kunci tetap tertangkap.
_BERAPA_RE = re.compile(r"\bb[er]*r?apa+\b")              # brapa, berapaa
_DIBAYAR_RE = re.compile(r"\bdi\s*ba+n?yar(?:in|i)?\b")   # dibayar, di bayar, di banyar
_BAYARAN_RE = re.compile(r"\bba+n?yar(?:an)?\b")          # bayar, banyar, bayaran


def _extract_emoji_score(raw: str) -> float:
    score = 0.0
    for ch in str(raw):
        if ch in POS_EMOJI:
            score += 1
        elif ch in NEG_EMOJI:
            score -= 1
        elif ch in MOCK_EMOJI:
            score -= 0.5
    return score


def _normalized_for_label(raw: str, slang: dict) -> str:
    """Versi ter-normalisasi slang TANPA stemming (jaga kata sentimen utuh)."""
    cleaned = clean_text(raw)
    normalized = normalize_text(cleaned, slang)
    normalized = _BERAPA_RE.sub("berapa", normalized)
    normalized = _DIBAYAR_RE.sub("bayar", normalized)
    normalized = _BAYARAN_RE.sub("bayar", normalized)
    return normalized


def label_comment(raw_text: str, slang: dict = None):
    """Kembalikan ``(label, score, detail)`` untuk satu komentar mentah."""
    if slang is None:
        slang = load_slang_dictionary()

    normalized = _normalized_for_label(raw_text, slang)
    tokens = normalized.split()

    score = 0.0
    hits = {"pos": [], "neg": [], "emoji": 0}

    # --- skor token dengan penanganan negasi (jendela 2 token ke belakang) ---
    for i, tok in enumerate(tokens):
        base = 0
        polarity = None
        if tok in NEG_WORDS:
            base = NEG_WORDS[tok]
            polarity = "neg"
        elif tok in POS_WORDS:
            base = POS_WORDS[tok]
            polarity = "pos"
        if polarity is None:
            continue
        negated = any(tokens[j] in NEGATION_WORDS for j in (i - 1, i - 2) if j >= 0)
        sign = -1 if polarity == "neg" else 1
        if negated:
            sign = -sign  # "bukan beban" -> positif; "tidak cerdas" -> negatif
        contrib = sign * base
        score += contrib
        hits["neg" if contrib < 0 else "pos"].append(tok + ("(neg)" if negated else ""))

    # --- skor frasa multi-kata ---
    for phrase, w in NEG_PHRASES.items():
        if phrase in normalized:
            score -= w
            hits["neg"].append(f"[{phrase}]")
    for phrase, w in POS_PHRASES.items():
        if phrase in normalized:
            score += w
            hits["pos"].append(f"[{phrase}]")

    # --- emoji (sekunder) ---
    emo = _extract_emoji_score(raw_text)
    hits["emoji"] = emo
    score += 0.5 * emo

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    detail = (
        f"neg={hits['neg']} pos={hits['pos']} emoji={emo}"
        if (hits["neg"] or hits["pos"] or emo)
        else "(tidak ada sinyal sentimen)"
    )
    return label, round(score, 2), detail


if __name__ == "__main__":
    tests = [
        "Sri Mulyani guru beban negara, di bayar berapa min?",
        "kasian bu sri di fitnah, ibu terbaik sehat selalu ❤️",
        "guru bukan beban negara, guru pahlawan tanpa tanda jasa",
        "ini AI kok",
        "Sri Mulyani",
        "halah bacot lu koruptor 🤬",
        "tetap semangat bu sri, sabar ya 🥺",
        "🤣🤣🤣",
        "❤️",
    ]
    slang = load_slang_dictionary()
    for t in tests:
        lab, sc, det = label_comment(t, slang)
        print(f"[{lab:>8}] {sc:>5}  {t}")
        print(f"           {det}")
