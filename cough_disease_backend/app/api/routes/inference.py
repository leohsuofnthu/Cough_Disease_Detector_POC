"""Inference API routes."""

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_inference_service
from app.services.inference_service import InferenceService
from app.services.schemas import HealthResponse, InferenceResponse


router = APIRouter(tags=["inference"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
  return HealthResponse()


@router.post("/infer", response_model=InferenceResponse)
async def infer(
  file: UploadFile = File(..., description="Audio file"),
  service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
  return await service.infer(file)

