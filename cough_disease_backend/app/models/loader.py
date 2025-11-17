"""Model loading utilities."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from loguru import logger

from app.core.config import settings
from app.models.predictor import BasePredictor, MockPredictor, TorchPredictor


# Resolve cough_detector code location inside container/host
APP_ROOT = Path(__file__).resolve().parents[2]  # typically /app
_candidates = [
  APP_ROOT / "cough_detector",
  APP_ROOT.parent / "cough_detector",
  Path("/app/cough_detector"),
]
for _p in _candidates:
  if _p.exists() and str(_p) not in sys.path:
    sys.path.append(str(_p))
    break


def _resolve_device(device_str: str) -> torch.device:
  device_str = device_str.lower()
  if device_str.startswith("cuda") and torch.cuda.is_available():
    return torch.device(device_str)
  return torch.device("cpu")


def _sanitize_state_dict(state_dict: dict) -> dict:
  sanitized = {}
  for key, value in state_dict.items():
    if key.startswith("module."):
      sanitized[key[len("module.") :]] = value
    else:
      sanitized[key] = value
  return sanitized


def _infer_classes(state_dict: dict) -> int:
  for key in ("fc_transfer.weight", "fc_transfer.bias"):
    if key in state_dict:
      return state_dict[key].shape[0]
  if "fc_audioset.weight" in state_dict:
    return state_dict["fc_audioset.weight"].shape[0]
  raise RuntimeError("Unable to infer number of classes from checkpoint")


def _default_labels_for_classes(count: int) -> List[str]:
  if count == 7:
    return [
      "Normal",
      "COPD",
      "Heart Disease",
      "Bronchiectasis",
      "Pneumonia",
      "Upper Respiratory Tract Infection",
      "Lower Respiratory Tract Infection",
    ]
  if count == 2:
    return ["No Cough", "Cough"]
  return [f"Class {idx}" for idx in range(count)]


def _load_checkpoint(path: Path, map_location: torch.device) -> Tuple[dict, int]:
  checkpoint = torch.load(path, map_location=map_location)
  for key in ("model_state_dict", "model", "state_dict"):
    if key in checkpoint and isinstance(checkpoint[key], dict):
      checkpoint = checkpoint[key]
      break
  if not isinstance(checkpoint, dict):
    raise RuntimeError("Checkpoint format not supported")
  checkpoint = _sanitize_state_dict(checkpoint)
  classes_num = _infer_classes(checkpoint)
  return checkpoint, classes_num


def _load_cough_detector(labels: Iterable[str], device: torch.device) -> TorchPredictor:
  try:
    module = importlib.import_module("models")
    config = importlib.import_module("config")
  except ModuleNotFoundError as exc:  # pragma: no cover - depends on external repo
    raise RuntimeError("cough_detector package not found. Ensure repo is on PYTHONPATH.") from exc

  weights_path = settings.model_weights_path
  state_dict, classes_num = _load_checkpoint(weights_path, map_location=device)

  labels_list = list(labels)
  if len(labels_list) != classes_num:
    logger.warning(
      "Configured label count (%s) does not match checkpoint (%s). Using defaults from checkpoint.",
      len(labels_list),
      classes_num,
    )
    labels_list = _default_labels_for_classes(classes_num)

  stage = "stage1" if classes_num <= 2 else "stage2"
  logger.info("Initializing CoughDetector (%s) with %s classes", stage, classes_num)

  model = module.CoughDetector(
    sample_rate=config.SAMPLE_RATE,
    window_size=config.WINDOW_SIZE,
    hop_size=config.HOP_SIZE,
    mel_bins=config.MEL_BINS,
    fmin=config.FMIN,
    fmax=config.FMAX,
    classes_num=classes_num,
    freeze_blocks=0,
    pretrained_path=None,
  )

  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  if missing:
    logger.warning("Missing keys when loading checkpoint: %s", missing)
  if unexpected:
    logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

  predictor = TorchPredictor(model=model, labels=labels_list, device=device)
  return predictor


def load_predictor() -> BasePredictor:
  labels = settings.class_labels
  if not labels:
    raise ValueError("No class labels configured.")

  device = _resolve_device(settings.device)
  weights_path = settings.model_weights_path

  if not weights_path.exists():
    logger.warning("Model weights not found at %s. Falling back to mock predictor.", weights_path)
    return MockPredictor(labels=labels)

  try:
    predictor = _load_cough_detector(labels, device)
    logger.info("Loaded cough detector weights from %s", weights_path)
    return predictor
  except Exception as exc:
    logger.exception("Failed to load model weights. Using mock predictor. Error: %s", exc)
    return MockPredictor(labels=labels)
