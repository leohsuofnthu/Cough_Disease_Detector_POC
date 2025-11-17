# Cough Detection with Transfer Learning

A professional, efficient repository for cough detection using 2-stage transfer learning with PANNs (Pretrained Audio Neural Networks).

## 🎯 Overview

This repository implements a streamlined 2-stage transfer learning pipeline for cough detection:

1. **Stage 1 (COUGHVID)**: Learn cough fundamentals from COUGHVID dataset
2. **Stage 2 (ICBHI)**: Multi-class disease classification on ICBHI dataset

## 🏗️ Architecture

- **Base Model**: PANNs CNN14 pretrained on AudioSet
- **Transfer Learning**: Progressive unfreezing strategy
- **Data Augmentation**: Mixup and SpecAugmentation
- **Loss Functions**: Binary Cross Entropy (Stage 1), NLL Loss (Stage 2)

## 📊 Training Strategy

| Stage | Dataset | Freeze | LR | Epochs | Description |
|-------|---------|--------|----|---------| ----------- |
| 1️⃣ | COUGHVID | First 2 conv blocks | 1e-4 | 10 | Learn cough fundamentals |
| 2️⃣ | ICBHI | First conv block only | 5e-5 | 15 | Multi-class disease classification |

## 🚀 Quick Start

### Installation

1. **Create & activate environment**
   ```bash
   conda create -n cough_detector python=3.8
   conda activate cough_detector
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Optional: installs ffmpeg if missing
   conda install -c conda-forge ffmpeg
   ```
3. **Download pretrained CNN14 weights**
   ```bash
   curl -L -o pretrained/Cnn14_mAP=0.431.pth \
     https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth?download=1
   ```
   The `pretrained/` directory includes a `.gitkeep` placeholder; place the downloaded file there before running any training scripts.
4. **Verify setup**
   ```bash
   python -c "import torch, librosa; print(torch.__version__, librosa.__version__)"
   python -c "from pydub import AudioSegment; print('pydub OK')"
   ```

### Data Preparation

1. **Download datasets**
   - COUGHVID (Stage 1): [Zenodo link](https://zenodo.org/records/4048312) → extract into `data/stage1/`
   - ICBHI (Stage 2): [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HT6PKI) → extract into `data/stage2/`

   Directory layout:
   ```
   data/
   ├── stage1/   # COUGHVID WAV + metadata CSV
   └── stage2/   # ICBHI WAV + annotation TXT files
   ```

2. **Preprocess audio**
   ```bash
   python convert_webm_to_wav.py --directory data/stage1 --sample_rate 32000
   python unified_data_processor.py
   ```

### Training (full pipeline)

```bash
python staged_training.py --stage all \
    --coughvid_path processed_data/stage1_coughvid.h5 \
    --icbhi_path processed_data/stage2_icbhi.h5 \
    --workspace ./workspace \
    --device cuda    # switch to cpu if needed
```

### Evaluation & Inference

```bash
# Single audio
python inference.py \
    --model_path ./workspace/checkpoints/stage2_icbhi/best_model.pth \
    --audio_path /path/to/audio.wav

# Batch directory
python inference.py \
    --model_path ./workspace/checkpoints/stage2_icbhi/best_model.pth \
    --audio_dir /path/to/audio_folder \
    --output_file predictions.json
```

## 📁 Repository Structure

```
cough_detector/
├── models.py / trainer.py / staged_training.py / losses.py
├── unified_data_processor.py / data_loader.py
├── config.py / pytorch_utils.py
├── inference.py / requirements.txt / README.md
├── data/ | processed_data/ | pretrained/ | workspace/
└── ...
```

## 🔧 Configuration & Monitoring

- Edit `config.py` for audio/training/model parameters.
- Training artifacts:
  - TensorBoard: `tensorboard --logdir ./workspace/logs`
  - Logs: `workspace/logs/{stage}/`
  - Checkpoints: `workspace/checkpoints/{stage}/`

## 📊 Expected Results

- Stage 1 (COUGHVID): ~85–90% accuracy (binary cough detection)
- Stage 2 (ICBHI): ~75–80% accuracy (multi-class disease detection)

## 🛠️ Advanced Usage

```python
from models import CoughDetector
model = CoughDetector(
    sample_rate=32000,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=7,
    freeze_blocks=0,
    pretrained_path="pretrained/Cnn14_mAP=0.431.pth"
)
```

```python
from trainer import CoughTrainer
trainer = CoughTrainer(stage='stage2_icbhi', workspace='./workspace', device='cuda')
best_acc = trainer.train(train_loader, val_loader)
```

## 🚨 Troubleshooting

1. CUDA OOM → reduce batch size in `config.py`
2. Dataset not found → confirm HDF5 paths
3. Import errors → ensure `conda activate cough_detector`
4. Slow training → use GPU, bump `num_workers`

## 📚 References

1. Kong, Q., et al. "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM TASLP 28 (2020).
2. Orlandic, L., et al. "The COUGHVID crowdsourcing dataset..." Scientific Data 8.1 (2021).
3. ICBHI 2017 Scientific Challenge on Respiratory Sound Analysis.