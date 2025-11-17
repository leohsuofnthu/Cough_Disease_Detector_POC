"""Audio helpers for inference."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as Fnn

from app.core.config import settings


def _load_with_torchaudio(tmp_path: str) -> Tuple[torch.Tensor, int]:
  waveform, sample_rate = torchaudio.load(tmp_path)
  if waveform.dtype != torch.float32:
    waveform = waveform.to(torch.float32)
  return waveform, sample_rate


def _load_with_soundfile(data: bytes) -> Tuple[torch.Tensor, int]:
  with sf.SoundFile(io.BytesIO(data)) as snd:
    samples = snd.read(dtype="float32")
    sample_rate = snd.samplerate

  samples = samples.astype(np.float32)
  if samples.ndim == 1:
    waveform = torch.from_numpy(samples).unsqueeze(0)
  else:
    waveform = torch.from_numpy(samples).T
  return waveform, sample_rate


def _load_with_ffmpeg(tmp_path: str, target_sr: int | None) -> Tuple[torch.Tensor, int]:
  fd, wav_path = tempfile.mkstemp(suffix=".wav")
  os.close(fd)
  cmd = [
    "ffmpeg",
    "-y",
    "-i",
    tmp_path,
    "-ac",
    "1",
  ]
  if target_sr:
    cmd.extend(["-ar", str(target_sr)])
  cmd.append(wav_path)

  proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if proc.returncode != 0:
    stderr = proc.stderr.decode("utf-8", errors="ignore")
    os.unlink(wav_path)
    raise RuntimeError(f"ffmpeg failed to decode audio: {stderr}")

  try:
    with open(wav_path, "rb") as wav_file:
      wav_bytes = wav_file.read()
    waveform, sample_rate = _load_with_soundfile(wav_bytes)
  finally:
    try:
      os.unlink(wav_path)
    except OSError:
      pass

  return waveform, sample_rate


def load_waveform(data: bytes, target_sr: int | None = None, filename: str | None = None) -> Tuple[torch.Tensor, int, float]:
  """Load raw audio bytes into a waveform tensor, tolerant of container formats."""

  suffix = Path(filename).suffix if filename else ""
  tmp_file = tempfile.NamedTemporaryFile(suffix=suffix or ".tmp", delete=False)
  try:
    tmp_file.write(data)
    tmp_file.flush()
    tmp_path = tmp_file.name
  finally:
    tmp_file.close()

  try:
    waveform, sample_rate = _load_with_torchaudio(tmp_path)
  except RuntimeError:
    try:
      waveform, sample_rate = _load_with_soundfile(data)
    except Exception:
      waveform, sample_rate = _load_with_ffmpeg(tmp_path, target_sr)
  finally:
    try:
      os.unlink(tmp_path)
    except OSError:
      pass

  if target_sr and target_sr != sample_rate:
    waveform = F.resample(waveform, orig_freq=sample_rate, new_freq=target_sr)
    sample_rate = target_sr

  if waveform.ndim == 2 and waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

  waveform = torch.clamp(waveform, -1.0, 1.0)
  max_val = waveform.abs().max()
  if max_val > 0:
    waveform = waveform / max_val

  orig_duration = waveform.shape[-1] / float(sample_rate)

  max_samples = int(settings.max_audio_seconds * sample_rate)
  if waveform.shape[-1] > max_samples:
    waveform = waveform[..., :max_samples]
  elif waveform.shape[-1] < max_samples:
    pad = max_samples - waveform.shape[-1]
    waveform = Fnn.pad(waveform, (0, pad))

  waveform = waveform.contiguous()

  duration = min(orig_duration, settings.max_audio_seconds)
  return waveform.squeeze(0), sample_rate, duration
