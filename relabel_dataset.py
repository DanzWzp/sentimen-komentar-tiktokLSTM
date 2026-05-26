"""
Pelabelan ulang komentar_tiktok -> komentar_tiktok_labeled.csv

Mengganti pelabelan TextBlob (yang membuat kelas netral membludak) dengan
pelabelan berbasis leksikon Bahasa Indonesia (lihat sentimen_labeling.py).
Menyimpan juga berkas review berisi skor & alasan agar mudah dikoreksi manual.
"""

import sys
from pathlib import Path

import pandas as pd

from sentimen_labeling import label_comment
from sentimen_preprocessing import load_slang_dictionary, preprocess

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
SRC = next(iter(sorted(BASE.glob("komentar_tiktok.*"))))
OUT = BASE / "komentar_tiktok_labeled.csv"
REVIEW = BASE / "review_label_sample.csv"
LABEL_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}


def main():
    print(f"Sumber data : {SRC.name}")
    df = pd.read_excel(SRC) if SRC.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(SRC)

    # samakan nama kolom
    cols = {c.lower().strip(): c for c in df.columns}
    comment_col = next((cols[c] for c in ("comment", "komentar", "text", "teks") if c in cols), None)
    user_col = next((cols[c] for c in ("username", "user", "nama", "akun") if c in cols), None)
    df = df.rename(columns={comment_col: "comment"})
    if user_col:
        df = df.rename(columns={user_col: "Username"})
    else:
        df["Username"] = "unknown"

    df["comment"] = df["comment"].astype(str).str.strip()
    df = df[df["comment"].str.lower().isin(["", "nan", "none"]) == False]
    df = df.dropna(subset=["comment"]).drop_duplicates(subset=["comment"]).reset_index(drop=True)
    print(f"Jumlah komentar unik: {len(df)}")

    slang = load_slang_dictionary()
    results = df["comment"].apply(lambda t: label_comment(t, slang))
    df["sentimen"] = results.apply(lambda r: r[0])
    df["skor"] = results.apply(lambda r: r[1])
    df["alasan"] = results.apply(lambda r: r[2])
    df["label_int"] = df["sentimen"].map(LABEL_TO_INT)

    # ringkasan distribusi
    dist = df["sentimen"].value_counts().reindex(["negative", "neutral", "positive"]).fillna(0).astype(int)
    total = int(dist.sum())
    print("\n=== Distribusi kelas hasil pelabelan ulang ===")
    for k in ["negative", "neutral", "positive"]:
        print(f"  {k:>8}: {dist[k]:>5}  ({dist[k] / total * 100:5.1f}%)")
    print(f"  {'TOTAL':>8}: {total:>5}")

    # simpan dataset berlabel (kolom inti + audit)
    df[["Username", "comment", "sentimen", "label_int", "skor", "alasan"]].to_csv(
        OUT, index=False, encoding="utf-8-sig"
    )
    print(f"\nDataset berlabel disimpan: {OUT.name}")

    # berkas review: contoh per kelas + kasus skor lemah (|skor|<=1) untuk dicek manual
    review = pd.concat([
        df.groupby("sentimen", group_keys=False).head(15),
        df[df["skor"].abs() <= 1].head(40),
    ]).drop_duplicates(subset=["comment"])
    review[["comment", "sentimen", "skor", "alasan"]].to_csv(REVIEW, index=False, encoding="utf-8-sig")
    print(f"Berkas review (untuk koreksi manual) disimpan: {REVIEW.name} ({len(review)} baris)")


if __name__ == "__main__":
    main()
