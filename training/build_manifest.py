"""Scan data/raw -> data/manifest.csv (path, species, breed, age).

Label dibaca otomatis dari NAMA FOLDER / FILE, contoh yang didukung:
  cat/siamese/adult/xxx.jpg      dog/golden_retriever/puppy/xxx.jpg
  kucing/persia/xxx.jpg          anjing/husky/xxx.jpg
  train/cat/xxx.jpg              Abyssinian_12.jpg (gaya Oxford-IIIT Pet)
Kata kunci usia: kitten/puppy/baby/young -> anak | adult/dewasa -> dewasa | senior/old/tua -> tua
Gambar tanpa label usia / jenis tetap dipakai (label itu diabaikan saat training).
"""
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "manifest.csv"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CAT = {"cat", "cats", "kucing", "feline"}
DOG = {"dog", "dogs", "anjing", "canine"}
AGE = {"kitten": "anak", "puppy": "anak", "baby": "anak", "young": "anak", "junior": "anak",
       "anak": "anak", "adult": "dewasa", "dewasa": "dewasa",
       "senior": "tua", "old": "tua", "tua": "tua", "elderly": "tua"}
STOP = {"data", "mining", "dataset", "raw", "images", "image", "img", "train", "training",
        "test", "testing", "val", "valid", "validation", "pets", "pet", "hewan",
        "klasifikasi", "jenis", "ras", "usia", "umur", "folder"}


def tokens(s: str):
    return [t for t in re.split(r"[^a-z]+", s.lower()) if t]


def species_of(toks):
    for t in toks:
        if t in CAT:
            return "cat"
        if t in DOG:
            return "dog"
    return None


by_hash, skipped, bad, dup = {}, Counter(), 0, 0
for p in sorted(RAW.rglob("*")):
    if p.suffix.lower() not in EXT:
        continue
    rel = p.relative_to(RAW)
    dirs = list(rel.parts[:-1])
    stem_toks = tokens(p.stem)

    species = age = breed = None
    for d in reversed(dirs):
        toks = tokens(d)
        species = species or species_of(toks)
        rest = [t for t in toks if t not in CAT | DOG | STOP]
        if rest and all(t in AGE for t in rest):
            # folder murni usia (mis. "Dewasa", "Puppy") -> label usia
            for t in rest:
                age = age or AGE.get(t)
        elif rest and breed is None:
            # folder jenis; kata seperti "Puppy" di nama jenis TIDAK dianggap usia
            breed = "_".join(rest)
    species = species or species_of(stem_toks)
    for t in stem_toks:
        age = age or AGE.get(t)

    # Gaya Oxford-IIIT Pet: "Maine_Coon_12.jpg" (huruf awal besar = kucing, kecil = anjing)
    m = re.match(r"^([A-Za-z]+(?:_[A-Za-z]+)*)_\d+$", p.stem)
    if m and breed is None:
        breed = m.group(1).lower()
        if species is None:
            species = "cat" if p.stem[0].isupper() else "dog"

    if species is None:
        skipped[str(rel.parent)] += 1
        continue
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception:
        bad += 1
        continue
    # Gambar yang sama bisa ada di 'Kucing' dan 'Klasifikasi Kucing': simpan satu saja,
    # pilih salinan yang labelnya paling lengkap (jenis/usia).
    h = hashlib.md5(p.read_bytes()).hexdigest()
    row = (rel.as_posix(), species, breed or "", age or "")
    score = bool(breed) + bool(age)
    if h in by_hash:
        dup += 1
        if score > by_hash[h][1]:
            by_hash[h] = (row, score)
    else:
        by_hash[h] = (row, score)

rows = [v[0] for v in by_hash.values()]

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "species", "breed", "age"])
    w.writerows(rows)

print(f"Total gambar valid : {len(rows)}   (rusak: {bad}, duplikat dibuang: {dup})")
print("Spesies            :", dict(Counter(r[1] for r in rows)))
print("Usia               :", dict(Counter(r[3] or '(tanpa label)' for r in rows)))
breeds = Counter((r[1], r[2]) for r in rows if r[2])
print(f"Jenis (breed)      : {len(breeds)} kelas")
for (sp, b), n in sorted(breeds.items()):
    print(f"   {sp:3s} {b:30s} {n}")
if skipped:
    print("\nFolder yang spesiesnya TIDAK terdeteksi (dilewati):")
    for k, v in skipped.most_common(10):
        print(f"   {k}  ({v} gambar)")
    print("-> ganti nama folder agar mengandung 'cat'/'kucing' atau 'dog'/'anjing'.")
print("\nManifest ditulis ke", OUT)
