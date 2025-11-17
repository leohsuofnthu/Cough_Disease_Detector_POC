"""Application configuration using Pydantic settings."""

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  """Central configuration object for the backend."""

  project_name: str = "Cough Disease Detector API"
  version: str = "0.1.0"
  api_v1_prefix: str = "/v1"
  log_level: str = "INFO"

  model_weights_path: Path = Path("models/cough_detector.pth")
  model_config_path: Optional[Path] = None
  device: str = "cpu"
  temp_dir: Path = Path("runtime_tmp")
  allowed_origins: List[str] = []
  max_upload_mb: int = 12
  class_labels: List[str] = [
    "Normal",
    "COPD",
    "Heart Disease",
    "Bronchiectasis",
    "Pneumonia",
    "Upper Respiratory Tract Infection",
    "Lower Respiratory Tract Infection",
  ]
  target_sample_rate: int = 32000
  max_audio_seconds: float = 10.0
  model_stage: str = "stage2"

  model_config = SettingsConfigDict(
    env_file=".env",
    env_prefix="BACKEND_",
    case_sensitive=False,
    protected_namespaces=("settings_",),
  )

  @property
  def max_upload_bytes(self) -> int:
    return self.max_upload_mb * 1024 * 1024

  @field_validator("allowed_origins", mode="before")
  @classmethod
  def parse_origins(cls, value: List[str] | str) -> List[str]:
    if isinstance(value, str):
      try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
          return [str(item) for item in parsed]
      except json.JSONDecodeError:
        pass
      return [item.strip() for item in value.split(",") if item.strip()]
    return value

  @field_validator("class_labels", mode="before")
  @classmethod
  def parse_labels(cls, value: List[str] | str) -> List[str]:
    if isinstance(value, str):
      try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
          return [str(item) for item in parsed]
      except json.JSONDecodeError:
        pass
      return [item.strip() for item in value.split(",") if item.strip()]
    return value


@lru_cache
def get_settings() -> Settings:
  return Settings()


settings = get_settings()
