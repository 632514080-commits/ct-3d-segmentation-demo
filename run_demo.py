from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from ctdemo.data import assert_no_subject_leakage, generate_dataset, load_manifest, normalize_hu
from ctdemo.metrics import segmentation_metrics
from ctdemo.model import TinyUNet3D, dice_loss


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


class CTDataset(Dataset):
    def __init__(self, manifest, split):
        self.cases = [c for c in manifest["cases"] if c["split"] == split]

    def __len__(self): return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        arrays = np.load(DATA / f"{case['case_id']}.npz")
        image = torch.from_numpy(normalize_hu(arrays["image"]))[None]
        mask = torch.from_numpy(arrays["mask"].astype(np.float32))[None]
        return image, mask, case["case_id"]


def seed_everything(seed=2026):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def largest_component(mask):
    labels, count = ndimage.label(mask)
    if count == 0: return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def traditional_baseline(volume):
    # Explainable hybrid baseline: HU threshold -> morphology -> largest component.
    candidate = volume > 95
    candidate = ndimage.binary_opening(candidate, iterations=1)
    return largest_component(candidate)


def train(manifest, epochs=3, device_name="cpu"):
    seed_everything()
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch environment")
    device = torch.device(device_name)
    model = TinyUNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = DataLoader(CTDataset(manifest, "train"), batch_size=2, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); running = 0.0
        for image, mask, _ in loader:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            # Lesion voxels are far below 1% of a volume. Weighting the positive
            # class prevents the trivial all-background solution in this tiny run.
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, mask, pos_weight=torch.tensor(50.0, device=device)
            )
            loss = dice_loss(logits, mask) + bce
            loss.backward(); optimizer.step(); running += loss.item()
        mean_loss = running / len(loader)
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch={epoch} loss={mean_loss:.4f} device={device}")
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history}, ARTIFACTS / "model.pt")
    return model.cpu(), history


def evaluate(model, manifest):
    rows = []
    model.eval()
    for image, target, case_id in DataLoader(CTDataset(manifest, "test"), batch_size=1):
        with torch.no_grad(): pred = torch.sigmoid(model(image)).numpy()[0, 0] > .5
        target_np = target.numpy()[0, 0].astype(bool)
        learned = segmentation_metrics(pred, target_np)
        raw = np.load(DATA / f"{case_id[0]}.npz")["image"]
        baseline = segmentation_metrics(traditional_baseline(raw), target_np)
        rows.append({"case_id": case_id[0], **{f"model_{k}": v for k, v in learned.items()},
                     **{f"baseline_{k}": v for k, v in baseline.items()}})
    with (ARTIFACTS / "test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {
        "n_test": len(rows),
        "model_mean_dice": float(np.mean([r["model_dice"] for r in rows])),
        "baseline_mean_dice": float(np.mean([r["baseline_dice"] for r in rows])),
        "worst_case": min(rows, key=lambda r: r["model_dice"])["case_id"],
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, summary


def export_and_benchmark(model):
    example = torch.zeros(1, 1, 32, 48, 48)
    scripted = torch.jit.trace(model.eval(), example)
    scripted.save(str(ARTIFACTS / "model.ts"))
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(5): scripted(example)
    latency = (time.perf_counter() - start) / 5 * 1000
    result = {"format": "TorchScript", "input_shape": list(example.shape), "cpu_latency_ms": latency,
              "model_size_mb": (ARTIFACTS / "model.ts").stat().st_size / 1024 ** 2}
    try:
        torch.onnx.export(model, example, ARTIFACTS / "model.onnx", input_names=["ct_volume"],
                          output_names=["lesion_logits"], opset_version=17)
        result["onnx_exported"] = True
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(ARTIFACTS / "model.onnx"), providers=["CPUExecutionProvider"])
        ort_output = ort_session.run(None, {"ct_volume": example.numpy()})[0]
        with torch.no_grad(): torch_output = model(example).numpy()
        result["onnx_max_abs_error"] = float(np.max(np.abs(ort_output - torch_output)))
        result["onnx_parity_passed"] = result["onnx_max_abs_error"] < 1e-4
    except Exception as exc:
        result["onnx_exported"] = False; result["onnx_note"] = str(exc)
    (ARTIFACTS / "deployment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def plot_example(model, manifest):
    case = next(c for c in manifest["cases"] if c["split"] == "test" and c["has_lesion"])
    arrays = np.load(DATA / f"{case['case_id']}.npz")
    image, target = arrays["image"], arrays["mask"]
    x = torch.from_numpy(normalize_hu(image))[None, None]
    with torch.no_grad(): pred = torch.sigmoid(model(x))[0, 0].numpy() > .5
    z = int(np.argmax(target.sum(axis=(1, 2))))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, title in zip(axes, ["CT slice", "Ground truth", "Model prediction"]): ax.axis("off"); ax.set_title(title)
    axes[0].imshow(image[z], cmap="gray", vmin=-200, vmax=300)
    axes[1].imshow(image[z], cmap="gray", vmin=-200, vmax=300); axes[1].imshow(target[z], alpha=.45, cmap="Reds")
    axes[2].imshow(image[z], cmap="gray", vmin=-200, vmax=300); axes[2].imshow(pred[z], alpha=.45, cmap="Blues")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "prediction.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="End-to-end synthetic 3D CT lesion segmentation demo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--device", choices=["cpu", "mps"], default="cpu",
        help="Training device. CPU is the reliable default because some PyTorch/MPS versions do not support Conv3d.",
    )
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    manifest_path = DATA / "manifest.json"
    if not manifest_path.exists(): generate_dataset(DATA)
    manifest = load_manifest(manifest_path); assert_no_subject_leakage(manifest)
    model, history = train(manifest, args.epochs, args.device)
    _, summary = evaluate(model, manifest)
    deployment = export_and_benchmark(model); plot_example(model, manifest)
    print(json.dumps({"dataset": manifest["dataset_version"], **summary, **deployment}, indent=2))


if __name__ == "__main__": main()

