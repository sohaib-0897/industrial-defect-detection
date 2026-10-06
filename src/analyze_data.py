"""Audit an explicit dataset root and produce leakage-aware frozen splits."""

import argparse
import csv
import hashlib
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from src.common import configure_logging, file_hash, write_json
from src.data import read_image

LOG = logging.getLogger(__name__)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALIASES = {
    "normal": "normal",
    "good": "normal",
    "ok": "normal",
    "defective": "defective",
    "defect": "defective",
    "bad": "defective",
    "ng": "defective",
}


def choose_splits(rows: list[dict], seed: int) -> list[dict]:
    labels = np.array([row["label"] for row in rows])
    groups = np.array([row["group"] for row in rows])
    if len(set(groups)) < 6:
        raise ValueError("Need at least six independent groups and both classes in each split")
    best, best_score = None, float("inf")
    # Candidate search balances each class without ever cutting a product/duplicate group.
    splitter = GroupShuffleSplit(n_splits=256, train_size=0.70, random_state=seed)
    for attempt, (train, remaining) in enumerate(splitter.split(labels, labels, groups)):
        if len(set(groups[remaining])) < 2:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + attempt)
        val_local, test_local = next(inner.split(remaining, labels[remaining], groups[remaining]))
        partitions = (train, remaining[val_local], remaining[test_local])
        if any(set(labels[p]) != {"normal", "defective"} for p in partitions):
            continue
        score = sum(
            abs(float(np.sum(labels[p] == label)) / np.sum(labels == label) - ratio)
            for p, ratio in zip(partitions, (0.70, 0.15, 0.15), strict=True)
            for label in ("normal", "defective")
        )
        if score < best_score:
            best, best_score = partitions, score
    if best is None:
        raise ValueError("Cannot form leakage-free splits containing both classes; inspect groups/add data")
    for indices, split in zip(best, ("train", "val", "test"), strict=True):
        for index in indices:
            rows[index]["split"] = split
    return rows


def audit(
    root: Path,
    output: Path,
    metadata: Path | None = None,
    group_regex: str | None = None,
    seed: int = 42,
    near_limit: int = 5000,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    if root.resolve() == output.resolve():
        raise ValueError("Audit output must differ from the dataset root")
    (output / "splits.csv").unlink(missing_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.resolve().is_relative_to(output.resolve()))
    images = [p for p in files if p.suffix.lower() in EXTENSIONS]
    report = {
        "status": "audit",
        "root": str(root.resolve()),
        "seed": seed,
        "total_files": len(files),
        "image_candidates": len(images),
        "corrupt": [],
        "unlabeled": [],
        "exact_duplicates": [],
        "near_duplicates": [],
        "assumptions": [
            "Folder aliases infer binary labels only; arbitrary labels require CSV.",
            "No filename grouping guessed. Supply CSV group or --group-regex.",
            "dHash is a conservative leakage heuristic, not proof of product identity.",
        ],
    }
    if not images:
        report["status"] = "blocked_no_images"
        write_json(output / "dataset_analysis.json", report)
        (output / "dataset_analysis.md").write_text(
            "No dataset images found. Real counts, labels, dimensions, duplicates and metrics are unavailable.\n",
            encoding="utf-8",
        )
        raise ValueError("No images found; supply the assessment dataset root")
    explicit = {}
    if metadata:
        with metadata.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = Path(row["path"]).as_posix()
                if key in explicit:
                    raise ValueError(f"Duplicate metadata path: {key}")
                if row["label"].lower() not in {"normal", "defective"} or not row.get("group", "").strip():
                    raise ValueError("Metadata requires path,label,group with canonical labels/nonempty group")
                explicit[key] = row
    pattern = re.compile(group_regex) if group_regex else None
    rows, hashes, dimensions, formats = [], [], Counter(), Counter()
    for path in images:
        relative = path.relative_to(root).as_posix()
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Image symlink escapes dataset root: {relative}")
        try:
            image = read_image(path)
            with Image.open(path) as original:
                formats[original.format] += 1
            dimensions[f"{image.width}x{image.height}"] += 1
        except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            report["corrupt"].append({"path": relative, "error": str(error)})
            continue
        if metadata:
            record = explicit.get(relative)
            label = record["label"].lower() if record else None
            group = record["group"] if record else relative
        else:
            inferred = {ALIASES[p.lower()] for p in path.relative_to(root).parts[:-1] if p.lower() in ALIASES}
            label = inferred.pop() if len(inferred) == 1 else None
            match = pattern.search(relative) if pattern else None
            if pattern and (match is None or not match.groups()):
                raise ValueError(f"Group regex must capture a group for every image: {relative}")
            group = match.group(1) if match else relative
        if label is None:
            report["unlabeled"].append(relative)
            continue
        pixels = np.asarray(image.resize((9, 8)).convert("L"))
        bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
        dhash = sum(int(bit) << index for index, bit in enumerate(bits))
        # Pixel digest finds lossless duplicates even across formats/metadata.
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        rows.append({"path": relative, "label": label, "group": group, "sha256": file_hash(path)})
        hashes.append((digest, dhash))
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(a: int, b: int) -> None:
        parents[find(b)] = find(a)

    seen_groups, seen_pixels = {}, {}
    for index, row in enumerate(rows):
        for seen, key in ((seen_groups, row["group"]), (seen_pixels, hashes[index][0])):
            if key in seen:
                merge(index, seen[key])
                if seen is seen_pixels:
                    report["exact_duplicates"].append([rows[seen[key]]["path"], row["path"]])
                    if rows[seen[key]]["label"] != row["label"]:
                        report.setdefault("conflicting_labels", []).append(row["path"])
            else:
                seen[key] = index
    report["near_duplicate_search"] = f"dHash Hamming <=2; exhaustive up to {near_limit} images"
    if len(rows) <= near_limit:
        for index, (_, dhash) in enumerate(hashes):
            for other in range(index):
                if hashes[index][0] != hashes[other][0] and (dhash ^ hashes[other][1]).bit_count() <= 2:
                    merge(index, other)
                    report["near_duplicates"].append([rows[other]["path"], rows[index]["path"]])
    else:
        report["near_duplicate_search"] = "Skipped: dataset exceeds configured exhaustive-search limit"
    for index, row in enumerate(rows):
        row["group"] = f"group-{find(index):06d}"
    report.update(
        {
            "readable_labeled_images": len(rows),
            "class_counts": dict(Counter(r["label"] for r in rows)),
            "dimensions": dict(dimensions),
            "formats": dict(formats),
            "independent_groups": len({r["group"] for r in rows}),
        }
    )
    write_json(output / "dataset_analysis.json", report)
    if report["unlabeled"] or report.get("conflicting_labels"):
        raise ValueError(
            "Unlabeled images/conflicting duplicate labels: resolve dataset_analysis.json before splitting"
        )
    rows = choose_splits(rows, seed)
    report["split_counts"] = {
        split: dict(Counter(r["label"] for r in rows if r["split"] == split)) for split in ("train", "val", "test")
    }
    report["status"] = "ready"
    report["excluded_corrupt_images"] = len(report["corrupt"])
    write_json(output / "dataset_analysis.json", report)
    with (output / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "group", "sha256", "split"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "dataset_analysis.md").write_text(
        "# Dataset audit\n\n"
        + f"Readable labeled images: {len(rows)}. Classes: {report['class_counts']}.\n\n"
        + f"Dimensions: {dict(dimensions)}. Formats: {dict(formats)}. Corrupt excluded: {len(report['corrupt'])}.\n\n"
        + f"Exact duplicate pairs: {len(report['exact_duplicates'])}; near pairs: {len(report['near_duplicates'])}. "
        + f"Independent merged groups: {report['independent_groups']}.\n\n"
        + f"Fixed seed {seed}; candidate group splits minimize deviation from 70/15/15 per class. {report['split_counts']}\n\n"
        + "Inspect near-duplicate pairs and source metadata before training; capture identity cannot be proven from pixels alone.\n",
        encoding="utf-8",
    )
    LOG.info("Dataset audit complete: %s", report["split_counts"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--group-regex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-limit", type=int, default=5000)
    args = parser.parse_args()
    configure_logging()
    audit(args.data, args.output, args.metadata, args.group_regex, args.seed, args.near_limit)


if __name__ == "__main__":
    main()
