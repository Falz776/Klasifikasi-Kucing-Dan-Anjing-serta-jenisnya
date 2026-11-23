"""Download dataset dari Google Drive ke data/raw."""
import zipfile
from pathlib import Path

import gdown

URL = "https://drive.google.com/drive/folders/1CG86UVWBhv3VRduiWx-6BXtEGoJ2tObb"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

print("Mengunduh dataset ... (folder Drive harus 'Anyone with the link')")
gdown.download_folder(url=URL, output=str(RAW), quiet=False, use_cookies=False, remaining_ok=True)

# Ekstrak otomatis jika isinya file .zip
for z in RAW.rglob("*.zip"):
    print("Ekstrak", z)
    with zipfile.ZipFile(z) as f:
        f.extractall(z.parent / z.stem)

print("\nSelesai. Catatan: gdown hanya mengambil maks. 50 file per folder.")
print("Jika datasetnya besar, unduh manual sebagai ZIP dari Drive lalu ekstrak ke:", RAW)
