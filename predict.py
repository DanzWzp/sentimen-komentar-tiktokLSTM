"""
Mesin prediksi sentimen untuk web interface.

Memuat model LSTM (.keras) + tokenizer + pemetaan label, lalu menjalankan
pipeline pra-pemrosesan yang IDENTIK dengan saat pelatihan sehingga prediksi
konsisten. Model hasil pelabelan ulang (``*_relabel*``) diutamakan; bila belum
ada, otomatis memakai model asli bawaan notebook.
"""

import json
import pickle
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"

# (model, tokenizer, params) — pasangan relabel diutamakan, lalu fallback asli.
_CANDIDATES = [
    ("model_lstm_relabel.keras", "tokenizer_relabel.pkl", "model_params_relabel.json"),
    ("model_terbaik_dengan_balancing.keras", "tokenizer.pkl", "model_params.json"),
]

_state = {"loaded": False}


def _resolve_artifacts():
    for model_f, tok_f, par_f in _CANDIDATES:
        if (ART / model_f).exists() and (ART / tok_f).exists():
            return model_f, tok_f, par_f
    raise FileNotFoundError(
        "Tidak ada model .keras + tokenizer di folder artifacts/. "
        "Jalankan retrain_model.py atau notebook terlebih dahulu."
    )


def load():
    """Muat model & artefak sekali saja (dipanggil saat web app start)."""
    if _state["loaded"]:
        return
    import tensorflow as tf  # impor di sini agar startup lain tetap cepat

    from sentimen_preprocessing import load_slang_dictionary, preprocess

    model_f, tok_f, par_f = _resolve_artifacts()
    _state["model"] = tf.keras.models.load_model(ART / model_f, compile=False)
    with open(ART / tok_f, "rb") as f:
        _state["tokenizer"] = pickle.load(f)

    params = json.loads((ART / par_f).read_text(encoding="utf-8")) if (ART / par_f).exists() else {}
    _state["maxlen"] = int(params.get("max_sequence_length", _state["model"].input_shape[-1] or 18))

    mapping_path = ART / "label_mapping.json"
    if mapping_path.exists():
        m = json.loads(mapping_path.read_text(encoding="utf-8"))
        _state["int_to_label"] = {int(k): v for k, v in m["int_to_label"].items()}
    else:
        _state["int_to_label"] = {0: "negative", 1: "neutral", 2: "positive"}

    _state["preprocess"] = preprocess
    _state["slang"] = load_slang_dictionary()
    _state["model_name"] = model_f
    _state["loaded"] = True


_LABEL_ID = {  # untuk styling/Bahasa Indonesia di UI
    "negative": {"id": "Negatif", "emoji": "😠"},
    "neutral": {"id": "Netral", "emoji": "😐"},
    "positive": {"id": "Positif", "emoji": "😊"},
}


def predict(text: str) -> dict:
    """Prediksi sentimen satu komentar. Mengembalikan dict siap-render."""
    load()
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    text_final = _state["preprocess"](text, _state["slang"])
    if not text_final.strip():
        return {
            "input": text,
            "text_final": "",
            "label": "neutral",
            "label_id": "Netral",
            "emoji": "😐",
            "confidence": 0.0,
            "probabilities": {"negative": 0.0, "neutral": 0.0, "positive": 0.0},
            "note": "Komentar kosong setelah pra-pemrosesan (mis. hanya emoji/tanda baca).",
        }

    seq = _state["tokenizer"].texts_to_sequences([text_final])
    pad = pad_sequences(seq, maxlen=_state["maxlen"], padding="post", truncating="post")
    probs = _state["model"].predict(pad, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = _state["int_to_label"].get(idx, "neutral")

    probabilities = {
        _state["int_to_label"].get(i, str(i)): float(probs[i]) for i in range(len(probs))
    }
    meta = _LABEL_ID.get(label, {"id": label, "emoji": ""})
    return {
        "input": text,
        "text_final": text_final,
        "label": label,
        "label_id": meta["id"],
        "emoji": meta["emoji"],
        "confidence": float(probs[idx]),
        "probabilities": probabilities,
        "note": "",
    }


def model_info() -> dict:
    load()
    return {"model_name": _state["model_name"], "maxlen": _state["maxlen"]}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for t in ["Sri Mulyani guru beban negara, dibayar berapa min?",
              "ibu sri mulyani terbaik, sehat selalu bu",
              "ini video AI atau asli ya?"]:
        out = predict(t)
        print(f"{out['label_id']:>8} ({out['confidence']:.0%})  <- {t}")
        print(f"          {out['probabilities']}")
