"""Post-evaluation visual diagnostics; never used for model or threshold selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.data import preprocessing, read_image
from src.model import load_checkpoint


def render(data: Path, checkpoint: Path, output: Path) -> None:
    torch.set_num_threads(2)
    model, metadata = load_checkpoint(checkpoint)
    records = json.loads((checkpoint.parent / "test_predictions.json").read_text(encoding="utf-8"))
    selected = []
    for true, pred in (("defective", "defective"), ("defective", "normal"), ("normal", "defective")):
        selected.extend([r for r in records if r["ground_truth"] == true and r["predicted_class"] == pred][:2])
    activations = []
    handle = model.layer4.register_forward_hook(lambda _module, _input, result: activations.append(result))
    output.mkdir(parents=True, exist_ok=True)
    try:
        for index, row in enumerate(selected):
            activations.clear()
            model.zero_grad(set_to_none=True)
            image = read_image(data / row["path"])
            tensor = preprocessing(contract=metadata["preprocessing"])(image).unsqueeze(0)
            logits = model(tensor)
            feature = activations[0]
            feature.retain_grad()
            logits[0, 1].backward()
            weights = feature.grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feature).sum(dim=1).relu()[0].detach().numpy()
            cam /= max(float(cam.max()), 1e-8)
            cam_image = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 640), Image.Resampling.BILINEAR)
            from src.data import letterbox

            base = letterbox(image)
            heat = (plt.get_cmap("jet")(np.asarray(cam_image) / 255)[..., :3] * 255).astype(np.uint8)
            overlay = Image.blend(base, Image.fromarray(heat), 0.35)
            sheet = Image.new("RGB", (448, 710), "white")
            sheet.paste(base, (0, 0))
            sheet.paste(overlay, (224, 0))
            ImageDraw.Draw(sheet).text(
                (5, 650),
                f"{row['path']} true={row['ground_truth']}\npred={row['predicted_class']} p(defect)={row['defect_probability']:.6f}\nLayer4 Grad-CAM; class=defective; coarse diagnostic",
                fill="black",
            )
            sheet.save(output / f"{index:02d}_{Path(row['path']).stem}.jpg")
    finally:
        handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/KolektorSDD2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/run-001/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports/ksdd2/gradcam"))
    args = parser.parse_args()
    render(args.data, args.checkpoint, args.output)
