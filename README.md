# Analisis Sentimen Komentar TikTok — Pelabelan Ulang + Web Interface

Revisi untuk dua permintaan dosen:

1. **Pelabelan ulang** `komentar_tiktok` agar **kelas netral tidak terlalu tinggi**.
2. **Web interface sederhana** untuk deteksi sentimen (model **LSTM + Word2Vec**).

---

## 1. Kenapa kelas netral dulu membludak?

Pelabelan lama (notebook) memakai **TextBlob** atas teks hasil **terjemahan Google**:

```
polarity < 0  -> negative
polarity > 0  -> positive
polarity == 0 -> neutral
```

Banyak komentar Bahasa Indonesia diterjemahkan menjadi **polaritas tepat 0,0**
(kata sentimen hilang saat translasi), sehingga **otomatis berlabel `neutral`**.
Akibatnya kelas netral mendominasi dan model jadi bias ke netral.

## 2. Solusi: pelabelan ulang berbasis leksikon

Pelabelan diganti dengan **leksikon + aturan Bahasa Indonesia** khusus isu ini
([sentimen_labeling.py](sentimen_labeling.py)), yang menilai **langsung** kata/frasa
Indonesia (termasuk slang, umpatan, idiom, tuduhan "dibayar berapa/buzzer",
"beban negara"), dengan **penanganan negasi** ("bukan beban" → positif) dan
**emoji**. Komentar yang memang beropini tidak lagi jatuh ke netral; netral
hanya untuk komentar faktual/penyebutan nama/ambigu.

**Distribusi kelas (1.743 komentar unik)** — lihat `figures/distribusi_kelas_relabel.png`:

| Kelas    | Jumlah | Proporsi |
|----------|-------:|---------:|
| Negatif  | 823    | 47,2 %   |
| Netral   | 755    | 43,3 %   |
| Positif  | 165    | 9,5 %    |

Netral **bukan lagi kelas dominan** (negatif kini terbesar), sesuai permintaan.

> Pelabelan ini bersifat **asisten** dan dapat dikoreksi manual. Berkas
> [review_label_sample.csv](review_label_sample.csv) berisi contoh per kelas +
> kasus skor lemah (`|skor| <= 1`) lengkap dengan **alasan**. Kasus sarkasme
> (mis. "pintar/pandai" yang mengejek) paling layak diperiksa ulang manual —
> cukup ubah kolom `sentimen`/`label_int` pada `komentar_tiktok_labeled.csv`.

## 3. Hasil melatih ulang model pada label baru

Model dilatih ulang ([retrain_model.py](retrain_model.py)) mengikuti pipeline
notebook (Word2Vec → tokenizer → embedding matrix → RandomOverSampler → LSTM):

| Metrik         | Model lama (label TextBlob) | Model baru (label leksikon) |
|----------------|----------------------------:|----------------------------:|
| Akurasi        | 0,633                       | **0,701**                   |
| F1-macro       | 0,566                       | **0,649**                   |

Artifacts baru disimpan **terpisah** (akhiran `_relabel`) di `artifacts/`, **tanpa
menimpa** model lama (berguna untuk perbandingan sebelum/sesudah di laporan).

---

## 4. Menjalankan Web Interface

```bash
python app.py
```

Buka **http://127.0.0.1:5000** → ketik/tempel komentar → **Analisis Sentimen**.
Hasil: label (Negatif/Netral/Positif), tingkat keyakinan, dan bar probabilitas
tiap kelas. Tersedia juga endpoint JSON:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"comment":"guru beban negara, dibayar berapa min?"}'
```

Web app memuat **model relabel** bila ada, dan otomatis **fallback** ke model
lama. Pra-pemrosesan input identik dengan saat pelatihan.

---

## 5. Berkas baru

| Berkas | Fungsi |
|--------|--------|
| [sentimen_preprocessing.py](sentimen_preprocessing.py) | Pipeline pra-pemrosesan bersama (clean → normalisasi slang JSON → token → stopword → stemming → lemma). Dipakai notebook, retrain, dan web. |
| [sentimen_labeling.py](sentimen_labeling.py) | Pelabel sentimen berbasis leksikon + aturan. |
| [relabel_dataset.py](relabel_dataset.py) | Membuat `komentar_tiktok_labeled.csv` + `review_label_sample.csv`. |
| [retrain_model.py](retrain_model.py) | Melatih ulang LSTM+Word2Vec pada label baru → artifacts `_relabel`. |
| [predict.py](predict.py) | Mesin prediksi (muat model + preprocessing). |
| [app.py](app.py) + [templates/index.html](templates/index.html) | Web interface Flask. |
| `komentar_tiktok_labeled.csv` | Dataset hasil pelabelan ulang. |
| `figures/distribusi_kelas_relabel.png` | Grafik distribusi kelas baru. |

## 6. Reproduksi penuh

```bash
pip install -r requirements.txt          # dependensi
python relabel_dataset.py                # 1) pelabelan ulang -> CSV berlabel
python retrain_model.py                  # 2) latih ulang -> artifacts _relabel
python app.py                            # 3) jalankan web interface
```

> **Catatan environment:** `urllib3` lama (1.25.x) rusak di Python 3.13 dan
> membuat `import tensorflow` gagal. Sudah diperbaiki dengan `urllib3>=2.2`.
> Kamus `combined_slang_words.txt` berformat **JSON**; pemuatan slang di
> notebook (Cell 3.1) sudah diperbaiki agar mem-parsing JSON (≈1.018 entri aktif,
> sebelumnya nyaris kosong).
