import csv

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.make_synthetic import create
from src.analyze_data import audit, choose_splits
from src.common import CLASS_TO_INDEX
from src.data import load_manifest, preprocessing, read_image


def test_preprocessing_is_deterministic_full_frame(image_bytes):
    image = read_image(image_bytes)
    first, second = preprocessing()(image), preprocessing()(image)
    assert first.shape == (3, 640, 224)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    # Wide input keeps both colored edge bands through letterboxing.
    edges = Image.new("RGB", (400, 100), "white")
    edges.paste("red", (0, 0, 30, 100))
    edges.paste("blue", (370, 0, 400, 100))
    result = preprocessing()(edges)
    assert result[0, 320, 3] > result[2, 320, 3]
    assert result[2, 320, 220] > result[0, 320, 220]


def test_grayscale_rgba_and_invalid():
    import io

    for mode in ("L", "RGBA"):
        stream = io.BytesIO()
        Image.new(mode, (10, 10)).save(stream, format="PNG")
        assert read_image(stream.getvalue()).mode == "RGB"
    with pytest.raises(OSError):
        read_image(b"not an image")


def test_training_augmentation_preserves_edge_defect_cues():
    image = Image.new("RGB", (230, 640), (30, 30, 30))
    image.paste("red", (0, 0, 8, 640))
    image.paste("blue", (222, 0, 230, 640))
    torch.manual_seed(42)
    transformed = preprocessing(training=True)(image)
    assert transformed.shape == (3, 640, 224)
    assert torch.isfinite(transformed).all()
    assert transformed[0, 320, 2] > transformed[2, 320, 2]
    assert transformed[2, 320, 221] > transformed[0, 320, 221]


def test_mapping():
    assert CLASS_TO_INDEX == {"normal": 0, "defective": 1}


def test_audit_duplicates_corruption_and_split_integrity(tmp_path):
    root, output = tmp_path / "data", tmp_path / "audit"
    create(root, per_class=20)
    original = root / "normal" / "SYNTHETIC_normal_000.png"
    (root / "normal" / "duplicate.png").write_bytes(original.read_bytes())
    near = read_image(original)
    red, green, blue = near.getpixel((0, 0))
    near.putpixel((0, 0), (min(red + 1, 255), green, blue))
    near.save(root / "normal" / "near.png")
    (root / "normal" / "corrupt.png").write_bytes(b"broken")
    report = audit(root, output)
    assert len(report["corrupt"]) == 1
    assert len(report["exact_duplicates"]) == 1
    assert report["near_duplicates"]
    rows = load_manifest(output / "splits.csv", root)
    copies = [
        r for r in rows if r["path"] in {"normal/duplicate.png", "normal/near.png", "normal/SYNTHETIC_normal_000.png"}
    ]
    assert len({r["split"] for r in copies}) == 1
    assert len({r["group"] for r in copies}) == 1
    initial = (output / "splits.csv").read_bytes()
    audit(root, output)
    assert initial == (output / "splits.csv").read_bytes()
    original.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="Dataset changed"):
        load_manifest(output / "splits.csv", root)


def test_group_split_and_insufficient_data():
    rows = [
        {"path": f"{label}_{i}_{j}", "label": label, "group": f"{label}_{i}"}
        for label in ("normal", "defective")
        for i in range(10)
        for j in range(2)
    ]
    split = choose_splits(rows, 42)
    for group in {row["group"] for row in split}:
        assert len({row["split"] for row in split if row["group"] == group}) == 1
    with pytest.raises(ValueError):
        choose_splits(rows[:2], 42)


def test_unlabeled_does_not_get_silent_label(tmp_path):
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(tmp_path / "unknown.png")
    with pytest.raises(ValueError, match="Unlabeled"):
        audit(tmp_path, tmp_path / "audit")


def test_metadata_groups(tmp_path):
    root = tmp_path / "data"
    create(root, per_class=20)
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "group"])
        writer.writeheader()
        for path in sorted(root.rglob("*.png")):
            index = int(path.stem.rsplit("_", 1)[1])
            writer.writerow(
                {
                    "path": path.relative_to(root).as_posix(),
                    "label": path.parent.name,
                    "group": f"{path.parent.name}-product-{index // 2}",
                }
            )
    report = audit(root, tmp_path / "audit", metadata)
    assert report["independent_groups"] <= 20
