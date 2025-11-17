# syntax=docker/dockerfile:1.6

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for torchaudio + soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts

RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir "setuptools>=68" wheel && \
    pip install --no-cache-dir \
      torch==2.2.2+cpu \
      torchaudio==2.2.2+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

COPY models ./models

ENV BACKEND_MODEL_WEIGHTS_PATH=/app/models/cough_detector.pth \
    BACKEND_DEVICE=cpu \
    BACKEND_LOG_LEVEL=INFO \
    BACKEND_ALLOWED_ORIGINS=* \
    BACKEND_TARGET_SAMPLE_RATE=32000 \
    BACKEND_MAX_AUDIO_SECONDS=10

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
