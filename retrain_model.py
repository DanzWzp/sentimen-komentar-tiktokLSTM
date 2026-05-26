"""
Latih ulang model LSTM + Word2Vec pada label hasil pelabelan ulang.

Pipeline mengikuti notebook (cabang "dengan_balancing" yang terbukti terbaik):
  preprocess -> split stratified -> Word2Vec(train) -> tokenizer + padding
  -> embedding matrix -> RandomOverSampler -> LSTM -> evaluasi -> simpan.

Artifacts disimpan dengan akhiran ``_relabel`` agar TIDAK menimpa model lama
(berguna untuk perbandingan sebelum/sesudah pada laporan). Web interface akan
memakai model relabel ini bila tersedia.
"""

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)
LABELED = BASE / "komentar_tiktok_labeled.csv"
LABEL_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def main():
    import tensorflow as tf
    from gensim.models import Word2Vec
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score, precision_recall_fscore_support)
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, Embedding,
                                         SpatialDropout1D)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical

    tf.random.set_seed(RANDOM_STATE)

    from sentimen_preprocessing import load_slang_dictionary, preprocess

    # ---------- 1. Muat label baru + preprocessing ----------
    print("Memuat label hasil pelabelan ulang ...")
    df = pd.read_csv(LABELED)
    slang = load_slang_dictionary()
    print(f"Preprocessing {len(df)} komentar (stemming Sastrawi, mohon tunggu) ...")
    df["text_final"] = df["comment"].astype(str).apply(lambda t: preprocess(t, slang))
    df = df[df["text_final"].str.len() > 0].drop_duplicates(subset=["text_final"]).reset_index(drop=True)
    df["label_int"] = df["sentimen"].map(LABEL_TO_INT)
    print(f"Korpus siap-model: {len(df)} komentar")
    print(df["sentimen"].value_counts().to_string())

    # ---------- 2. Split stratified ----------
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["label_int"], random_state=RANDOM_STATE
    )
    X_train_text = train_df["text_final"].tolist()
    X_test_text = test_df["text_final"].tolist()
    X_train_tokens = [t.split() for t in X_train_text]
    y_train_int = train_df["label_int"].to_numpy()
    y_test_int = test_df["label_int"].to_numpy()

    # ---------- 3. Word2Vec pada data latih ----------
    print("Melatih Word2Vec ...")
    w2v = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5,
                   min_count=1, workers=4, sg=1, epochs=30, seed=RANDOM_STATE)

    # ---------- 4. Tokenizer + padding ----------
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    seq_len = [len(s) for s in X_train_seq if len(s) > 0]
    MAXLEN = int(np.clip(np.percentile(seq_len, 95), 5, 100))
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAXLEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAXLEN, padding="post", truncating="post")

    # ---------- 5. Embedding matrix dari Word2Vec ----------
    VOCAB = len(tokenizer.word_index) + 1
    EMB = w2v.vector_size
    emb_matrix = np.random.normal(0.0, 0.05, (VOCAB, EMB)).astype(np.float32)
    emb_matrix[0] = 0.0
    covered = 0
    for word, idx in tokenizer.word_index.items():
        if word in w2v.wv:
            emb_matrix[idx] = w2v.wv[word]
            covered += 1
    print(f"vocab={VOCAB-1}, maxlen={MAXLEN}, cakupan Word2Vec={covered/(VOCAB-1):.1%}")

    # ---------- 6. Balancing (RandomOverSampler) ----------
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    Xb, yb = ros.fit_resample(X_train_pad, y_train_int)
    yb_cat = to_categorical(yb, num_classes=3)
    y_test_cat = to_categorical(y_test_int, num_classes=3)

    # ---------- 7. Arsitektur LSTM (identik notebook) ----------
    model = Sequential([
        Embedding(input_dim=VOCAB, output_dim=EMB, weights=[emb_matrix],
                  input_length=MAXLEN, trainable=True),
        SpatialDropout1D(0.2),
        LSTM(128, dropout=0.4),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]
    print("Melatih LSTM ...")
    model.fit(Xb, yb_cat, validation_split=0.2, epochs=20, batch_size=32,
              callbacks=callbacks, verbose=2)

    # ---------- 8. Evaluasi ----------
    y_prob = model.predict(X_test_pad, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_test_int, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test_int, y_pred, average="macro", zero_division=0)
    print(f"\n=== Evaluasi data uji ===\nakurasi={acc:.4f}  f1_macro={f1:.4f}")
    print(classification_report(y_test_int, y_pred,
          target_names=["negative", "neutral", "positive"], zero_division=0))

    # ---------- 9. Simpan artifacts (akhiran _relabel) ----------
    model.save(ART / "model_lstm_relabel.keras")
    with open(ART / "tokenizer_relabel.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    w2v.save(str(ART / "word2vec_relabel.model"))
    params = {"vocab_size": VOCAB, "embedding_dim": EMB, "max_sequence_length": MAXLEN,
              "lstm_units": 128, "dense_units": 64, "dropout_rate": 0.4,
              "learning_rate": 1e-3, "batch_size": 32, "epochs": 20, "num_classes": 3}
    (ART / "model_params_relabel.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (ART / "label_mapping.json").write_text(json.dumps(
        {"label_to_int": LABEL_TO_INT, "int_to_label": {str(k): v for k, v in INT_TO_LABEL.items()}},
        indent=2), encoding="utf-8")
    pd.DataFrame([{"eksperimen": "relabel_dengan_balancing", "accuracy": round(acc, 4),
                   "f1_macro": round(f1, 4), "precision_macro": round(p, 4),
                   "recall_macro": round(r, 4)}]).to_csv(
        ART / "hasil_metrics_relabel.csv", index=False)
    print("\nArtifacts relabel tersimpan di:", ART)


if __name__ == "__main__":
    main()
