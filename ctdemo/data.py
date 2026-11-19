from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CaseMeta:
    case_id: str
    subject_id: str
    split: str
    seed: int
    has_lesion: bool
    lesion_radius: float


def _ellipsoid(shape: tuple[int, int, int], center, radius) -> np.ndarray:
    z, y, x = np.indices(shape)
    cz, cy, cx = center
    rz, ry, rx = radius
    return ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1


def make_case(seed: int, shape=(32, 48, 48), has_lesion=True):
    """Generate a CT-like volume in HU and a binary lesion mask."""
    rng = np.random.default_rng(seed)
    body = _ellipsoid(shape, (16, 24, 24), (14, 20, 20))
    organ = _ellipsoid(shape, (16, 24, 24), (9, 13, 14))
    volume = np.full(shape, -1000.0, dtype=np.float32)
    volume[body] = rng.normal(-80, 18, body.sum())
    volume[organ] = rng.normal(55, 14, organ.sum())
    mask = np.zeros(shape, dtype=np.uint8)
    radius = 0.0
    if has_lesion:
        radius = float(rng.uniform(2.8, 5.2))
        center = (rng.integers(12, 21), rng.integers(19, 30), rng.integers(18, 30))
        lesion = _ellipsoid(shape, center, (radius, radius * .9, radius * 1.1)) & organ
        volume[lesion] = rng.normal(125, 12, lesion.sum())
        mask[lesion] = 1
    volume += rng.normal(0, 8, shape).astype(np.float32)
    return volume, mask, radius


def generate_dataset(root: Path, n_subjects=30, seed=2026) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    subject_ids = [f"mouse-{i:03d}" for i in range(n_subjects)]
    rng.shuffle(subject_ids)
    n_train, n_val = int(.7 * n_subjects), int(.15 * n_subjects)
    split_by_subject = {
        sid: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, sid in enumerate(subject_ids)
    }
    records = []
    # Two scans per subject intentionally force subject-level splitting.
    for subject_index, subject_id in enumerate(sorted(subject_ids)):
        for visit in range(2):
            case_seed = seed + subject_index * 10 + visit
            has_lesion = (subject_index + visit) % 5 != 0
            volume, mask, radius = make_case(case_seed, has_lesion=has_lesion)
            case_id = f"{subject_id}-v{visit + 1}"
            split = split_by_subject[subject_id]
            np.savez_compressed(root / f"{case_id}.npz", image=volume, mask=mask)
            records.append(asdict(CaseMeta(case_id, subject_id, split, case_seed, has_lesion, radius)))
    manifest = {"dataset_version": "synthetic-ct-v1", "generator_seed": seed, "cases": records}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    manifest["content_hash"] = hashlib.sha256(manifest_bytes).hexdigest()[:16]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_no_subject_leakage(manifest: dict) -> None:
    seen: dict[str, str] = {}
    for case in manifest["cases"]:
        old = seen.setdefault(case["subject_id"], case["split"])
        if old != case["split"]:
            raise ValueError(f"Data leakage: {case['subject_id']} appears in {old} and {case['split']}")


def normalize_hu(volume: np.ndarray) -> np.ndarray:
    return np.clip((volume + 200.0) / 500.0, 0.0, 1.0).astype(np.float32)

