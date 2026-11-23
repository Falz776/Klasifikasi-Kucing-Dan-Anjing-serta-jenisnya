"""Definisi model & konstanta yang dipakai bersama oleh train.py dan export_onnx.py."""
import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 224
SPECIES = ["cat", "dog"]
AGE_CLASSES = ["anak", "dewasa", "tua"]
AGE_RANGES = {
    "anak": "sekitar < 1 tahun",
    "dewasa": "sekitar 1 - 7 tahun",
    "tua": "sekitar > 7 tahun",
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PetNet(nn.Module):
    """Satu backbone, tiga 'kepala': spesies, jenis (ras), usia."""

    def __init__(self, n_breed: int, n_age: int = len(AGE_CLASSES), pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        feat = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.drop = nn.Dropout(0.3)
        self.head_species = nn.Linear(feat, len(SPECIES))
        self.head_breed = nn.Linear(feat, max(n_breed, 1))
        self.head_age = nn.Linear(feat, n_age)

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.head_species(f), self.head_breed(f), self.head_age(f)


class ExportWrapper(nn.Module):
    """Dibungkus agar ONNX menerima gambar 0..1 (NCHW) dan mengeluarkan probabilitas."""

    def __init__(self, net: PetNet):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        s, b, a = self.net(x)
        return s.softmax(-1), b.softmax(-1), a.softmax(-1)
