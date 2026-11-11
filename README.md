# Academic Detector – Backend Edition

Proyek ini mengembangkan file HTML awal menjadi frontend + API backend. Ini bukan replika internal Turnitin dan tidak mengakses database Turnitin.

## Fitur
- FastAPI REST API (`/api/analyze`, `/api/health`).
- Ekstraksi statistik linguistik dan analisis per kalimat.
- AI classifier opsional dari model Hugging Face yang di-fine-tune khusus untuk label `human` dan `ai`.
- Similarity TF-IDF terhadap dokumen `.txt` yang operator masukkan ke `corpus/`.
- UI HTML asli dipertahankan; tombol analisis mengirim teks ke backend.

## Menjalankan secara lokal

Disarankan Python 3.11 atau 3.12.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Buka `http://127.0.0.1:8000/`.

## Mengaktifkan AI classifier

AI score sengaja tidak dibuat-buat. Sebelum model tersedia, UI menampilkan `N/A` dan tetap menyediakan analisis linguistik.

1. Siapkan dataset berizin dalam CSV `text,label` (0=human, 1=AI); lihat `training/README_DATASET.md`.
2. Instal dependensi training (sudah tercantum di requirements).
3. Jalankan:

```bash
python training/train_classifier.py --csv training/dataset.csv --base-model xlm-roberta-base --output models/ai-classifier
```

4. Atur `AI_MODEL_PATH=./models/ai-classifier`, lalu restart API.

Model dasar XLM-RoBERTa bukan detektor AI siap pakai. Ia baru menjadi classifier setelah fine-tuning dengan dataset yang representatif dan evaluasi yang benar.

## Mengaktifkan similarity

Letakkan dokumen referensi yang boleh digunakan sebagai `.txt` di folder `corpus/`, termasuk subfolder bila perlu. Lalu panggil:

```bash
curl -X POST http://127.0.0.1:8000/api/corpus/reload
```

API hanya membandingkan dengan corpus tersebut. Similarity bukan putusan plagiarisme; kutipan, daftar pustaka, template, dan sumber yang sah perlu ditinjau manusia.

## API

- `GET /api/health`: status backend, model AI, jumlah dokumen corpus.
- `POST /api/analyze`: JSON `{"text":"...","include_similarity":true,"top_k":5}`.
- `GET /api/corpus/status`: status corpus.
- `POST /api/corpus/reload`: muat ulang corpus lokal.
- `GET /docs`: dokumentasi OpenAPI interaktif.

## Catatan produksi

Gunakan HTTPS, autentikasi, pembatasan ukuran dan laju request, penghapusan file sementara, kebijakan retensi dokumen, audit akses, dan penyimpanan terenkripsi. Batasi CORS ke domain frontend yang benar. Jangan menyimpan teks akademik sensitif ke log. Untuk penggunaan publik, tambahkan autentikasi dan otorisasi pada endpoint admin corpus.
