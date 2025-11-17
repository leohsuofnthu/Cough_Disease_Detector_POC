"""Pydantic schemas for API responses."""

from datetime import datetime
from typing import Dict

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
  status: str = Field(default="ok")
  detail: str = Field(default="ready")


class InferenceResponse(BaseModel):
  label: str
  confidence: float = Field(ge=0, le=1)
  probabilities: Dict[str, float]
  duration: float = Field(description="Duration in seconds", ge=0)
  timestamp: datetime

