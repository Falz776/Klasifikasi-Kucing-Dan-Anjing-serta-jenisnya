"""Export models/best.pt -> models/pet_classifier.onnx (+ labels.json)."""
import json
from pathlib import Path

import numpy as np
import torch

from common import AGE_RANGES, IMG_SIZE, ExportWrapper, PetNet

MODELS = Path(__file__).resolve().parents[1] / "models"


def export():
    ck = torch.load(MODELS / "best.pt", map_location="cpu")
    labels = ck["labels"]
    labels["age_ranges"] = AGE_RANGES
    net = PetNet(len(labels["breeds"]), len(labels["age_classes"]), pretrained=False)
    net.load_state_dict(ck["state"])
    model = ExportWrapper(net).eval()

    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    out = MODELS / "pet_classifier.onnx"
    kwargs = dict(input_names=["image"], output_names=["species", "breed", "age"],
                  dynamic_axes={"image": {0: "batch"}, "species": {0: "batch"},
                                "breed": {0: "batch"}, "age": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:  # torch < 2.5
        torch.onnx.export(model, dummy, str(out), **kwargs)

    (MODELS / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

    # verifikasi: hasil ONNX harus sama dengan PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = [t.numpy() for t in model(dummy)]
    got = sess.run(None, {"image": dummy.numpy()})
    diff = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    print(f"ONNX tersimpan: {out}  (selisih maks vs PyTorch = {diff:.2e})")


if __name__ == "__main__":
    export()
