# Cough Disease Detector Backend

Backend inference service for the cough disease detector. Implements best practices for serving
pretrained audio classification models built in `cough_detector/`.

## Highlights
- FastAPI app with async inference endpoint at `POST /v1/infer`
- Multipart ICBHI-stage inference (7-class disease taxonomy)
- Structured logging with Loguru
- Configurable model paths via environment variables or `.env`
- Background warmup and health-check endpoints
- Decoupled service layers: API, services, model loading, utilities

## Getting Started
1. Create a Python virtual environment (3.9+ recommended).
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Export model weights (for example into `models/cough_detector.pth`).
4. Provide configuration via environment or `.env` (see `app/core/config.py`).
5. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Container Usage

Build the image:
```bash
docker build -t cough-disease-backend .
```

Run backend only with weights mounted:
```bash
docker run --rm -p 8000:8000 \
  -e BACKEND_MODEL_WEIGHTS_PATH=/weights/cough_detector.pth \
  -v $(pwd)/models:/weights:ro \
  cough-disease-backend
```

Or use backend-only Compose:
```bash
docker-compose up --build
```

Full stack (backend + Expo web) lives in the repo root Compose file:
```bash
cd ..
docker compose up --build
```
- Ensure the `cough_detector/` training code and `cough_disease_backend/models/cough_detector.pth` weights exist so the backend container can import the model implementation and load the checkpoint.

## Environment Variables
- `MODEL_WEIGHTS_PATH` (required): path to `.pth` model weights.
- `MODEL_CONFIG_PATH` (optional): JSON/YAML describing model architecture.
- `DEVICE` (optional): `cpu` or `cuda`.
- `TEMP_DIR` (optional): path for temporary audio storage.
- `CLASS_LABELS` (optional): comma-separated override for ICBHI classes (defaults baked in).
- `TARGET_SAMPLE_RATE` (optional): target sampling rate for preprocessing (default 32000).
- `MAX_AUDIO_SECONDS` (optional): max duration clip for padding/truncation (default 10s).
- `MODEL_STAGE` (optional): stage identifier (`stage1` / `stage2`) for logging/documenting model lineage.

## Project Structure
```
cough_disease_backend/
├── app/
│   ├── api/
│   │   ├── dependencies.py
│   │   └── routes/
│   │       └── inference.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── models/
│   │   ├── loader.py
│   │   └── predictor.py
│   ├── services/
│   │   ├── inference_service.py
│   │   └── schemas.py
│   ├── utils/
│   │   ├── audio.py
│   │   └── filesystem.py
│   └── main.py
├── models/
│   └── README.md (weight placement instructions)
├── scripts/
│   └── run_dev.sh
├── pyproject.toml
└── README.md
```

## Next Steps
- Implement actual Torch model loading logic in `app/models/loader.py` leveraging
  `cough_detector/models.py` code.
- Harden security (rate limiting, auth) if exposing publicly.
- Add monitoring / tracing (OpenTelemetry, Prometheus) for production.
- Containerize using Docker and orchestrate with your infrastructure of choice.
