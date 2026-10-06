from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.ksdd2 import mask_label, official_split


def test_annotation_drives_label(tmp_path):
    path = tmp_path / "arbitrary_GT.png"
    values = np.zeros((30, 10), dtype=np.uint8)
    Image.fromarray(values).save(path)
    assert mask_label(path, (10, 30))[0] == "normal"
    values[15, 5] = 255
    Image.fromarray(values).save(path)
    assert mask_label(path, (10, 30))[:2] == ("defective", 1)
    with pytest.raises(ValueError, match="mismatch"):
        mask_label(path, (30, 10))
    values[0, 0] = 127
    Image.fromarray(values).save(path)
    with pytest.raises(ValueError, match="values"):
        mask_label(path, (10, 30))


def test_official_test_is_never_repartitioned():
    rows = [
        {"path": f"{split}/{label}/{i}", "official_split": split, "label": label}
        for split, count in (("train", 50), ("test", 10))
        for label in ("normal", "defective")
        for i in range(count)
    ]
    first = official_split([dict(r) for r in rows])
    assert first == official_split([dict(r) for r in rows])
    assert all(r["split"] == "test" for r in first if r["official_split"] == "test")
    assert sum(r["split"] == "val" for r in first) == 20
    assert sum(r["split"] == "val" and r["label"] == "defective" for r in first) == 10


@pytest.mark.skipif(not Path("data/KolektorSDD2").exists(), reason="Public dataset not downloaded")
def test_real_manifest_and_official_counts():
    from collections import Counter

    from src.data import load_manifest

    rows = load_manifest(Path("reports/ksdd2/splits.csv"), Path("data/KolektorSDD2"))
    assert len(rows) == 3335
    assert Counter(r["label"] for r in rows if r["split"] == "test") == {"normal": 894, "defective": 110}
    assert all(r["split"] == "test" for r in rows if r["official_split"] == "test")
