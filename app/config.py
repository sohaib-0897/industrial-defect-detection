import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(default_factory=lambda: Path(os.getenv("MODEL_PATH", "artifacts/run-001/best.pt")))
    backend: str = field(default_factory=lambda: os.getenv("INFERENCE_BACKEND", "pytorch"))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024))))
    threads: int = field(default_factory=lambda: int(os.getenv("INFERENCE_THREADS", "2")))

    def __post_init__(self) -> None:
        if self.backend not in {"pytorch", "onnx"} or self.max_upload_bytes < 1 or self.threads < 1:
            raise ValueError("Invalid inference settings")
