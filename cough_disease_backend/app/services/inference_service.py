"""Inference service orchestrating audio preprocessing and model prediction."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import HTTPException, UploadFile, status
from loguru import logger

from app.core.config import settings
from app.models.predictor import BasePredictor
from app.services.schemas import InferenceResponse
from app.utils.audio import load_waveform


class InferenceService:
  def __init__(self, predictor: BasePredictor) -> None:
    self.predictor = predictor
    self.max_upload_bytes = settings.max_upload_bytes
    self.target_sample_rate = settings.target_sample_rate

  async def warm(self) -> None:
    logger.info("Warming inference service")
    await asyncio.to_thread(self.predictor.warm)

  async def infer(self, upload_file: UploadFile) -> InferenceResponse:
    data = await upload_file.read()
    await upload_file.close()

    if not data:
      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    if len(data) > self.max_upload_bytes:
      raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Audio file too large")

    waveform, sample_rate, duration = await asyncio.to_thread(
      load_waveform, data, self.target_sample_rate, upload_file.filename
    )

    prediction = await asyncio.to_thread(self.predictor.predict_proba, waveform, sample_rate)

    return InferenceResponse(
      label=prediction.label,
      confidence=prediction.confidence,
      probabilities=prediction.probabilities,
      duration=duration,
      timestamp=datetime.utcnow(),
    )
