# Pet Classifier — Kucing vs Anjing + Jenis + Perkiraan Usia (ONNX)

Satu model (EfficientNet-B0, multi-task) dengan 3 output: **spesies**, **jenis/ras**, **kelompok usia** (anak / dewasa / tua).
Training pakai PyTorch → diekspor ke **ONNX** → dijalankan backend FastAPI (onnxruntime) → UI web.

```
pet-classifier/
├── data/raw/            <- dataset hasil download
├── data/manifest.csv    <- dibuat otomatis
├── models/              <- pet_classifier.onnx, labels.json, best.pt (hasil training)
├── training/            <- download_data, build_manifest, train, export_onnx
├── backend/app.py       <- API /predict
└── frontend/index.html  <- tampilan web
```

## 1. Setup (VS Code terminal, dari folder pet-classifier)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements-train.txt
```
GPU NVIDIA? Install torch CUDA dulu dari https://pytorch.org/get-started/locally/ (opsional, CPU juga bisa tapi lambat).

## 2. Download dataset
```bash
python training/download_data.py
```
`gdown` hanya mengambil maks. 50 file per folder. Kalau dataset lebih besar, di Google Drive klik kanan folder →
**Download** (ZIP) → ekstrak ke `data/raw/`.

## 3. Buat manifest (label otomatis dari nama folder)
```bash
python training/build_manifest.py
```
Baca ringkasannya: jumlah spesies, daftar jenis, label usia. Contoh struktur yang terbaca:
`cat/siamese/adult/x.jpg`, `dog/husky/puppy/x.jpg`, `kucing/persia/x.jpg`, atau `Abyssinian_12.jpg`.
Kata usia: `kitten/puppy/baby/young`=anak, `adult/dewasa`=dewasa, `senior/old/tua`=tua.
Kalau ada folder yang spesiesnya tidak terdeteksi, ganti nama foldernya (harus mengandung cat/kucing/dog/anjing).

## 4. Training + export ONNX
```bash
python training/train.py --epochs 12 --batch 32
```
Selesai → `models/pet_classifier.onnx` dan `models/labels.json`.
(Windows error dataloader? tambahkan `--workers 0`.)

## 5. Jalankan aplikasi (lokal)
```bash
uvicorn backend.app:app --reload --port 8010
```
Buka **http://127.0.0.1:8010** (frontend disajikan oleh backend yang sama).
Dokumentasi API otomatis: http://127.0.0.1:8010/docs

## 6. Online (opsional)
- Cepat/sementara: `ngrok http 8000` atau Cloudflare Tunnel.
- Permanen: deploy ke Render / Railway / Hugging Face Spaces (Docker). Perintah start: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
  Yang perlu di-upload: `backend/`, `frontend/`, `models/pet_classifier.onnx`, `models/labels.json`, dan requirements
  (untuk server cukup: fastapi, uvicorn, python-multipart, onnxruntime, numpy, pillow).

## Catatan penting
- **Usia butuh label usia.** Kalau dataset hanya berisi kucing/anjing + jenis, head usia otomatis tidak dilatih dan UI memberi tahu. Tambahkan folder usia (mis. `cat/siamese/kitten/`) agar aktif.
- Usia diprediksi sebagai 3 kelompok (anak < 1 th, dewasa 1–7 th, tua > 7 th), bukan angka tahun, karena label berbasis foto tidak akurat untuk angka pasti.
- Model hanya mengenal kucing & anjing; foto objek lain tetap dipaksa jadi salah satunya (UI menandai keyakinan rendah).

## Deploy online
- **Render**: lihat `render.yaml` (build: `pip install -r requirements.txt`).
- **Vercel**: `api/index.py` + `vercel.json`. Import repositori di vercel.com, tanpa framework, lalu Deploy.
  Halaman web di `/`, API di `/api/predict` dan `/api/health`.
- `requirements.txt` = paket server (ringan). `requirements-train.txt` = paket untuk training.
