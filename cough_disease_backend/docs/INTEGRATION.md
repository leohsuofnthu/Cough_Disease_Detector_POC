# Connecting the Backend to the Expo Frontend

1. **Start the backend**
   ```bash
   cd cough_disease_backend
   pip install -e .
   BACKEND_MODEL_WEIGHTS_PATH=/absolute/path/to/cough_detector.pth uvicorn app.main:app --reload
   ```

2. **Expose endpoint to frontend**
   - Default inference endpoint: `POST http://localhost:8000/v1/infer`
   - Request payload: multipart form with field name `file` (audio data)
   - Response contains the 7 ICBHI classes (Normal, COPD, Heart Disease, Bronchiectasis, Pneumonia, URTI, LRTI)
   - When running via Docker Compose, the Expo app uses `EXPO_PUBLIC_API_URL=http://backend:8000/v1/infer`

3. **Update Expo app** (`cough_disease_ui_pro/utils/api.ts`)
   ```ts
   const API_URL = "http://localhost:8000/v1/infer"; // or your deployed URL
   ```

4. **CORS**
   - Configure `BACKEND_ALLOWED_ORIGINS=http://localhost:8081,http://localhost:19006` (Expo web/dev URLs)
   - Or leave `*` for development only.

5. **Testing**
   - Use `curl`:
     ```bash
     curl -X POST http://localhost:8000/v1/infer \
       -F "file=@/path/to/sample.wav"
     ```
   - Expect JSON response `{ "label": ..., "confidence": ..., "probabilities": {"Normal": 0.3, ...} }`

6. **Docker Compose (full stack)**
   - From repo root: `docker compose up --build`
   - `backend` service serves FastAPI at `http://localhost:8000`
   - `frontend` service serves Expo web dev at `http://localhost:3000`

7. **Deployment considerations**
- Containerize backend with gunicorn/uvicorn workers.
- Configure HTTPS and authentication before exposing publicly.
