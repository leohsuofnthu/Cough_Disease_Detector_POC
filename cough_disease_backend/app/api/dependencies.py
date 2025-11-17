"""Dependency providers for FastAPI."""

from functools import lru_cache

from app.models.loader import load_predictor
from app.services.inference_service import InferenceService


@lru_cache
def get_inference_service() -> InferenceService:
  predictor = load_predictor()
  return InferenceService(predictor)

