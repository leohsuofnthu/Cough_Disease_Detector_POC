"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import inference
from app.api.dependencies import get_inference_service
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.schemas import HealthResponse


setup_logging(settings.log_level)

app = FastAPI(title=settings.project_name, version=settings.version)

origins = settings.allowed_origins or ["*"]
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
  service = get_inference_service()
  await service.warm()


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
  return HealthResponse(detail="Service ready")


app.include_router(inference.router, prefix=settings.api_v1_prefix)

