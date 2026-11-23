"""FastAPI + ONNX Runtime. Jalankan dari root proyek:  uvicorn backend.app:app --reload"""
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pet_classifier.onnx"
LABELS_PATH = ROOT / "models" / "labels.json"

if not MODEL_PATH.exists():
    raise SystemExit("Model belum ada. Jalankan training/train.py dulu (atau taruh pet_classifier.onnx + labels.json di folder models/).")

labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
SIZE = labels["img_size"]
BREED_SPECIES = np.array(labels["breed_species"])
NAMES_ID = {"cat": "Kucing", "dog": "Anjing"}

app = FastAPI(title="Pet Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def preprocess(img: Image.Image) -> np.ndarray:
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    img = img.crop((l, t, l + SIZE, t + SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


@app.get("/health")
def health():
    return {"status": "ok", "breeds": len(labels["breeds"]), "age_available": labels["age_available"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Ukuran gambar maksimal 10 MB.")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "File bukan gambar yang valid.")

    sp, br, ag = (o[0] for o in session.run(None, {"image": preprocess(img)}))
    si = int(sp.argmax())
    species = labels["species"][si]

    # jenis: hanya dari ras yang cocok dengan spesies hasil prediksi
    breeds = []
    if len(BREED_SPECIES):
        p = br[: len(BREED_SPECIES)] * (BREED_SPECIES == si)
        if p.sum() > 0:
            p = p / p.sum()
            for i in np.argsort(-p)[:3]:
                if p[i] > 0:
                    breeds.append({"name": pretty(labels["breeds"][i]), "prob": float(p[i])})

    age = None
    if labels["age_available"]:
        ai = int(ag.argmax())
        k = labels["age_classes"][ai]
        age = {"group": k, "range": labels["age_ranges"][k], "prob": float(ag[ai]),
               "all": {c: float(ag[i]) for i, c in enumerate(labels["age_classes"])}}

    return {"species": {"id": species, "name": NAMES_ID[species], "prob": float(sp[si])},
            "breeds": breeds, "age": age}


# Frontend disajikan dari server yang sama -> buka http://localhost:8000
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
