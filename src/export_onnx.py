import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.common import configure_logging, file_hash, write_json
from src.data import preprocessing, read_image
from src.model import load_checkpoint


def export(checkpoint_path: Path, samples: list[Path], threads: int = 2) -> dict:
    torch.set_num_threads(threads)
    model, checkpoint = load_checkpoint(checkpoint_path)
    path = checkpoint_path.with_suffix(".onnx")
    generator = torch.Generator().manual_seed(42)
    height = 640 if "224x640" in checkpoint["preprocessing"] else 224
    example = torch.randn((1, 3, height, 224), generator=generator)
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = [example, torch.randn((2, 3, height, 224), generator=generator)]
    inputs.extend(
        preprocessing(contract=checkpoint["preprocessing"])(read_image(sample)).unsqueeze(0) for sample in samples
    )
    errors = []
    comparisons = []
    with torch.inference_mode():
        for tensor in inputs:
            expected = model(tensor).numpy()
            actual = session.run(["logits"], {"image": tensor.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            errors.append(float(np.max(np.abs(actual - expected))))
            expected_prob = torch.from_numpy(expected).softmax(dim=1)[:, 1].numpy()
            actual_prob = torch.from_numpy(actual).softmax(dim=1)[:, 1].numpy()
            np.testing.assert_allclose(actual_prob, expected_prob, rtol=1e-4, atol=1e-5)
            agreement = np.array_equal(actual_prob >= checkpoint["threshold"], expected_prob >= checkpoint["threshold"])
            if not agreement:
                raise ValueError("PyTorch/ONNX predicted classes disagree")
            comparisons.append(
                {
                    "pytorch_probabilities": expected_prob.tolist(),
                    "onnx_probabilities": actual_prob.tolist(),
                    "classes_agree": bool(agreement),
                    "max_abs_probability_error": float(np.max(np.abs(actual_prob - expected_prob))),
                }
            )
    report = {
        "architecture": "resnet18",
        "class_names": checkpoint["class_names"],
        "preprocessing": checkpoint["preprocessing"],
        "threshold": checkpoint["threshold"],
        "model_version": checkpoint["model_version"],
        "checkpoint_sha256": file_hash(checkpoint_path),
        "onnx_sha256": file_hash(path),
        "parity_max_abs_logit_error": max(errors),
        "parity_inputs": len(inputs),
        "sample_images": [str(s) for s in samples],
        "probability_and_class_parity": comparisons,
        "opset": 17,
        "atol": 1e-5,
        "rtol": 1e-4,
    }
    write_json(path.with_suffix(".json"), report)
    logging.getLogger(__name__).info("ONNX export verified: %s", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/run-001/best.pt"))
    parser.add_argument("--samples", type=Path, nargs="*", default=[])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    configure_logging()
    export(args.checkpoint, args.samples, args.threads)
