import io

import pytest
import torch
from PIL import Image

from src.common import CLASS_NAMES
from src.model import build_model


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (300, 180), (90, 110, 130)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "fixture.pt"
    model = build_model(pretrained=False)
    # Fixed fixture prediction exercises real preprocessing/model/threshold integration.
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor([1.0, 0.0]))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "resnet18",
            "class_names": list(CLASS_NAMES),
            "preprocessing": "letterbox224-imagenet-v1",
            "threshold": 0.4,
            "model_version": "test-fixture",
        },
        path,
    )
    return path
