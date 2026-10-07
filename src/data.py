import csv
import io
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from src.common import CLASS_TO_INDEX, file_hash

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESSING = "letterbox224x640-imagenet-v2"
Image.MAX_IMAGE_PIXELS = 20_000_000


def read_image(source: Path | bytes) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            if image.format not in {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}:
                raise ValueError("Unsupported image format")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images are unsupported")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")


def letterbox(image: Image.Image) -> Image.Image:
    return ImageOps.pad(image, (224, 640), method=Image.Resampling.BILINEAR, color=(124, 116, 104))


def preprocessing(
    training: bool = False, contract: str = PREPROCESSING, rotation_degrees: float = 0
) -> transforms.Compose:
    steps = []
    if training:
        # Full-frame resize; no crop, flips, blur, or erasing that could remove defects.
        if rotation_degrees:
            steps.append(transforms.RandomRotation(degrees=rotation_degrees, expand=True, fill=(124, 116, 104)))
        steps.append(transforms.ColorJitter(brightness=0.10, contrast=0.10))
    if contract == PREPROCESSING:
        steps.append(transforms.Lambda(letterbox))
    elif contract == "letterbox224-imagenet-v1":
        steps.append(
            transforms.Lambda(
                lambda image: ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR, color=(124, 116, 104))
            )
        )
    else:
        raise ValueError("Unsupported preprocessing contract")
    steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(steps)


def load_manifest(path: Path, root: Path, verify: bool = True) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty split manifest")
    groups: dict[str, str] = {}
    paths: set[str] = set()
    for row in rows:
        target = (root / row["path"]).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("Image path escapes dataset root")
        if row["label"] not in CLASS_TO_INDEX or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid label/split in manifest")
        if row["path"] in paths:
            raise ValueError("Repeated image path in manifest")
        paths.add(row["path"])
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("Group leakage across splits")
        if verify and file_hash(target) != row["sha256"]:
            raise ValueError(f"Dataset changed: {row['path']}; regenerate audited splits")
        if verify and row.get("mask"):
            mask = (root / row["mask"]).resolve()
            if not mask.is_relative_to(root.resolve()) or file_hash(mask) != row["mask_sha256"]:
                raise ValueError(f"Dataset annotation changed: {row['path']}")
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != set(CLASS_TO_INDEX):
            raise ValueError(f"{split} must contain both classes")
    return rows


class InspectionDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, training: bool = False, rotation_degrees: float = 0):
        self.rows, self.root = rows, root
        self.transform = preprocessing(training, rotation_degrees=rotation_degrees)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        return self.transform(read_image(self.root / row["path"])), CLASS_TO_INDEX[row["label"]]
