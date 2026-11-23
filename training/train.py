"""Latih model multi-tugas (spesies + jenis + usia) lalu export ke ONNX."""
import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from common import AGE_CLASSES, IMG_SIZE, MEAN, SPECIES, STD, PetNet
from export_onnx import export

ROOT = Path(__file__).resolve().parents[1]
RAW, MODELS = ROOT / "data" / "raw", ROOT / "models"


class PetDS(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(RAW / r["path"]).convert("RGB")
        return self.tf(img), int(r["y_sp"]), int(r["y_br"]), int(r["y_ag"])


def masked_ce(logits, target, weight=None):
    """Cross-entropy yang mengabaikan sampel tanpa label (target = -1)."""
    m = target >= 0
    if m.sum() == 0:
        return logits.sum() * 0
    return F.cross_entropy(logits[m], target[m], weight=weight, label_smoothing=0.05)


def split(df, val_frac=0.15, seed=42):
    rng = random.Random(seed)
    val_idx = []
    for _, g in df.groupby(["species", "breed"]):
        idx = list(g.index)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac)) if len(idx) >= 5 else 0
        val_idx += idx[:n_val]
    val = df.loc[val_idx]
    return df.drop(val.index), val


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    hit, tot = np.zeros(3), np.zeros(3)
    yb_all, pb_all = [], []
    for x, ys, yb, ya in loader:
        outs = model(x.to(dev))
        for k, (o, y) in enumerate(zip(outs, (ys, yb, ya))):
            y = y.to(dev)
            m = y >= 0
            hit[k] += (o.argmax(1)[m] == y[m]).sum().item()
            tot[k] += m.sum().item()
            if k == 1 and m.any():
                yb_all += y[m].cpu().tolist()
                pb_all += o.argmax(1)[m].cpu().tolist()
    acc = hit / np.maximum(tot, 1)
    # akurasi jenis dirata-rata per kelas (adil untuk data tidak seimbang)
    if yb_all:
        yb_a, pb_a = np.array(yb_all), np.array(pb_all)
        rec = [(pb_a[yb_a == c] == c).mean() for c in np.unique(yb_a)]
        acc[1] = float(np.mean(rec))
    return acc, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-per-breed", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", dev)

    df = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False)
    cnt = Counter(df.loc[df.breed != "", "breed"])
    breeds = sorted(b for b, n in cnt.items() if n >= a.min_per_breed)
    b2i = {b: i for i, b in enumerate(breeds)}
    breed_species = [SPECIES.index(df.loc[df.breed == b, "species"].value_counts().idxmax()) for b in breeds]

    df["y_sp"] = df.species.map({s: i for i, s in enumerate(SPECIES)})
    df["y_br"] = df.breed.map(b2i).fillna(-1).astype(int)
    df["y_ag"] = df.age.map({s: i for i, s in enumerate(AGE_CLASSES)}).fillna(-1).astype(int)
    age_available = bool((df.y_ag >= 0).sum() >= 30)
    if not age_available:
        print("!! Label usia tidak ditemukan -> head usia TIDAK dilatih (lihat README).")
        df["y_ag"] = -1
    print(f"Gambar: {len(df)} | breed: {len(breeds)} | berlabel usia: {(df.y_ag >= 0).sum()}")

    train_df, val_df = split(df)
    tf_train = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)), T.RandomHorizontalFlip(),
        T.ColorJitter(0.25, 0.25, 0.25, 0.03), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_val = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])
    dl_tr = DataLoader(PetDS(train_df, tf_train), a.batch, shuffle=True, num_workers=a.workers,
                       pin_memory=dev == "cuda", drop_last=True)
    dl_va = DataLoader(PetDS(val_df, tf_val), a.batch, num_workers=a.workers)

    # jenis tidak seimbang (mis. 700+ vs 20 foto) -> beri bobot lebih untuk kelas langka
    bc = train_df.loc[train_df.y_br >= 0, "y_br"].value_counts()
    w = np.ones(max(len(breeds), 1), dtype=np.float32)
    for c, n in bc.items():
        w[int(c)] = 1.0 / np.sqrt(n)
    w = w / w.mean()
    breed_w = torch.tensor(w, device=dev)

    model = PetNet(len(breeds)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr))
    use_amp = dev == "cuda"  # AMP hanya untuk NVIDIA
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    labels = {"species": SPECIES, "breeds": breeds, "breed_species": breed_species,
              "age_classes": AGE_CLASSES, "age_available": age_available, "img_size": IMG_SIZE}
    MODELS.mkdir(exist_ok=True)
    best = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {ep}/{a.epochs}")
        for x, ys, yb, ya in pbar:
            x, ys, yb, ya = x.to(dev), ys.to(dev), yb.to(dev), ya.to(dev)
            with torch.cuda.amp.autocast(enabled=use_amp):
                s, b, g = model(x)
                loss = masked_ce(s, ys) + masked_ce(b, yb, breed_w) + 0.7 * masked_ce(g, ya)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        acc, tot = evaluate(model, dl_va, dev)
        print(f"  val acc  spesies={acc[0]:.3f}  jenis(rata2 per kelas)={acc[1]:.3f}  usia={acc[2]:.3f}  (n={tot.astype(int).tolist()})")
        score = acc[0] + (acc[1] if tot[1] else 0) + (acc[2] if tot[2] else 0)
        if score > best:
            best = score
            torch.save({"state": model.state_dict(), "labels": labels}, MODELS / "best.pt")
            print("  model terbaik disimpan")

    export()


if __name__ == "__main__":
    main()
