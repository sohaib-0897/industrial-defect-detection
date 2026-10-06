from typing import Literal

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    predicted_class: Literal["normal", "defective"]
    confidence: float = Field(ge=0, le=1)
    defect_probability: float = Field(ge=0, le=1)
    model_version: str
    decision_threshold: float = Field(ge=0, le=1)
    inference_time_ms: float = Field(ge=0)
