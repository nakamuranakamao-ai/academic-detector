# Dataset training

Sediakan `dataset.csv` dengan kolom:

- `text`: teks asli berbahasa Indonesia (atau bahasa target).
- `label`: `0` untuk tulisan manusia, `1` untuk teks AI.

Panduan:
- Gunakan data yang legal dan memiliki izin penggunaan.
- Pisahkan sumber/penulis antara train dan validation agar tidak terjadi data leakage.
- Seimbangkan kelas, topik, panjang teks, tingkat pendidikan, dan ragam bahasa.
- Jangan menganggap akurasi satu dataset sebagai bukti generalisasi.
- Evaluasi false positive pada tulisan manusia; laporkan precision, recall, F1, confusion matrix, dan kalibrasi.
- Jangan menggabungkan teks dari satu dokumen ke train dan validation sekaligus.
