"""Model predictor abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from loguru import logger

from app.core.config import settings


@dataclass
class Prediction:
  label: str
  confidence: float
  probabilities: Dict[str, float]


class BasePredictor:
  """Defines the interface predictors must implement."""

  def warm(self) -> None:  # pragma: no cover - simple interface
    """Optionally perform warmup to reduce first-request latency."""

  def predict_proba(self, waveform: torch.Tensor, sample_rate: int) -> Prediction:
    raise NotImplementedError


class TorchPredictor(BasePredictor):
  """Torch based predictor using a classification head that outputs logits."""

  def __init__(self, model: torch.nn.Module, labels: Iterable[str], device: torch.device) -> None:
    self.model = model.to(device)
    self.labels = list(labels)
    self.device = device
    self.model.eval()
    logger.info("Torch predictor initialized on {device}", device=device)

  def warm(self) -> None:
    with torch.no_grad():
      dummy_length = int(settings.target_sample_rate * settings.max_audio_seconds)
      dummy = torch.zeros(1, dummy_length, device=self.device)
      _ = self.model(dummy)

  def predict_proba(self, waveform: torch.Tensor, sample_rate: int) -> Prediction:
    if waveform.ndim == 2:
      if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
      else:
        waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
      waveform = waveform.unsqueeze(0)

    waveform = waveform.to(self.device)
    with torch.no_grad():
      output = self.model(waveform)
      clipwise = output.get('clipwise_output')
      if clipwise is None:
        raise RuntimeError('Model did not return clipwise_output')
      if clipwise.ndim == 1:
        clipwise = clipwise.unsqueeze(0)
      probs = torch.softmax(clipwise, dim=-1)[0].cpu().numpy()

    if len(probs) != len(self.labels):
      raise ValueError("Model output does not match label count")

    prob_dict = {label: float(prob) for label, prob in zip(self.labels, probs)}
    top_label = max(prob_dict, key=prob_dict.get)
    return Prediction(label=top_label, confidence=prob_dict[top_label], probabilities=prob_dict)


class MockPredictor(BasePredictor):
  """Fallback predictor used when weights are unavailable."""

  def __init__(self, labels: List[str]) -> None:
    self.labels = labels
    logger.warning("Using mock predictor - replace with real weights for production.")

  def predict_proba(self, waveform: torch.Tensor, sample_rate: int) -> Prediction:
    base = {label: 1.0 / len(self.labels) for label in self.labels}
    top_label = self.labels[0]
    return Prediction(label=top_label, confidence=base[top_label], probabilities=base)
