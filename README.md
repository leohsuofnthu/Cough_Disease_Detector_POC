# Cough Disease Detector – Monorepo

<div align="center">
  <img src="ui.png" alt="Cough Disease Detector UI" />
</div>

This repository hosts the complete cough-detection application:

1. **Model training** (`cough_detector/`): PyTorch pipeline that produces a PANNs-based classifier.
2. **Backend API** (`cough_disease_backend/`): FastAPI service that loads the trained model.
3. **Frontend app** (`cough_disease_ui_pro/`): React Native/Expo client for recording audio and displaying predictions.

## Project Structure

```
cough_disease_detector/
├── cough_detector/           # Model training code
├── cough_disease_backend/    # FastAPI + model serving
├── cough_disease_ui_pro/     # Expo/React Native client
└── docker-compose.yml        # Full stack deployment
```

## End-to-End Workflow

1. **Train the model**
   ```bash
   cd cough_detector
   # Follow README.md inside this folder to preprocess data and run:
   python staged_training.py --stage all \
       --coughvid_path processed_data/stage1_coughvid.h5 \
       --icbhi_path processed_data/stage2_icbhi.h5 \
       --workspace ./workspace \
       --device cuda   # switch to cpu if needed
   ```

2. **Publish the best checkpoint to the backend**
   - Copy the Stage 2 best checkpoint into the backend models directory:
     ```
     Source: cough_detector/workspace/checkpoints/stage2_icbhi/best_model.pth
     Target: cough_disease_backend/models/cough_detector.pth
     ```

3. **Run the full application**
   ```bash
   cd /path/to/cough_disease_detector
   docker compose up --build
   ```
   This command builds/starts the backend API and frontend UI containers defined in the root `docker-compose.yml`.

## Notes
- Keep the model filename as `cough_detector.pth` so the backend loader (`cough_disease_backend/app/models/loader.py`) can find it without code changes.
- If you retrain the model, repeat step 2 to refresh the backend weights before running `docker compose`.
- Each subproject still has its own README with deeper setup/testing instructions; use this root README as the high-level orchestrator.

