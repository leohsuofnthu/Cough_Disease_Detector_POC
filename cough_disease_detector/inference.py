"""
Inference script for cough detection models
Ensures same preprocessing as training pipeline
"""
import os
import torch
import numpy as np
import librosa
import soundfile as sf
from models import CoughDetector
from config import SAMPLE_RATE, WINDOW_SIZE, HOP_SIZE, MEL_BINS, FMIN, FMAX
import argparse
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoughInference:
    """Cough detection inference with consistent preprocessing"""
    
    def __init__(self, model_path: str, stage: Optional[str] = None, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Audio processing parameters (matching training)
        self.sample_rate = SAMPLE_RATE
        self.window_size = WINDOW_SIZE
        self.hop_size = HOP_SIZE
        self.mel_bins = MEL_BINS
        self.fmin = FMIN
        self.fmax = FMAX
        self.max_duration = 10.0
        self.max_samples = int(self.max_duration * self.sample_rate)
        
        # Load model and resolve actual stage
        self.model, resolved_stage = self._load_model(model_path, stage)
        self.stage = resolved_stage
        self.model.eval()
        self.class_names = self._get_class_names()
    
    def _get_class_names(self):
        if self.stage == 'stage1':
            return ['No Cough', 'Cough']
        return [
            'Normal', 'COPD', 'Heart Disease', 'Bronchiectasis',
            'Pneumonia', 'Upper Respiratory Tract Infection',
            'Lower Respiratory Tract Infection'
        ]
    
    def _load_model(self, model_path: str, stage_hint: Optional[str]):
        """Load trained model"""
        logger.info(f"🔄 Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Determine number of classes based on stage hint or checkpoint
        if stage_hint in ('stage1', 'stage2'):
            stage = stage_hint
            classes_num = 2 if stage == 'stage1' else 7
        elif 'stage1' in os.path.basename(model_path):
            stage = 'stage1'
            classes_num = 2
        elif 'stage2' in os.path.basename(model_path):
            stage = 'stage2'
            classes_num = 7
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                fc_transfer_weight = checkpoint['model_state_dict']['fc_transfer.weight']
                classes_num = fc_transfer_weight.shape[0]
                stage = 'stage1' if classes_num == 2 else 'stage2'
            else:
                raise ValueError("Cannot determine model type from checkpoint")
        
        # Create model
        model = CoughDetector(
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            hop_size=self.hop_size,
            mel_bins=self.mel_bins,
            fmin=self.fmin,
            fmax=self.fmax,
            classes_num=classes_num,
            freeze_blocks=0,  # No freezing for inference
            pretrained_path=None  # Load from checkpoint instead
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        logger.info(f"✅ Model loaded successfully ({stage}, {classes_num} classes)")
        
        return model, stage
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file with same settings as training
        This matches the preprocessing in unified_data_processor.py
        """
        try:
            # Load audio with librosa (same as training)
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize to [-1, 1] range (same as training)
            waveform = self._normalize_audio(waveform)
            
            # Pad or truncate to fixed length (same as training)
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            else:
                waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))
            
            # Ensure float32 precision (same as training)
            waveform = waveform.astype(np.float32)
            
            # Validate audio quality
            if np.isnan(waveform).any() or np.isinf(waveform).any():
                raise ValueError("Invalid audio data")
            
            return waveform
            
        except Exception as e:
            logger.error(f"Error preprocessing {audio_path}: {e}")
            return None
    
    def _normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range (matching training)"""
        waveform = np.clip(waveform, -1.0, 1.0)
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict on a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with prediction results
        """
        logger.info(f"🔄 Processing: {audio_path}")
        
        # Preprocess audio
        waveform = self.preprocess_audio(audio_path)
        if waveform is None:
            return {'error': 'Failed to preprocess audio'}
        
        # Convert to tensor
        waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output_dict = self.model(waveform_tensor)
            predictions = output_dict['clipwise_output']
            
            # Convert to probabilities
            if self.stage == 'stage1':
                # Binary classification - use sigmoid
                probabilities = torch.sigmoid(predictions).cpu().numpy()[0]
            else:
                # Multi-class classification - use softmax
                probabilities = torch.softmax(predictions, dim=-1).cpu().numpy()[0]
            
            # Get predicted class
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Create results
            results = {
                'audio_path': audio_path,
                'predicted_class': predicted_class,
                'predicted_class_name': self.class_names[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, probabilities)
                }
            }
            
            logger.info(f"✅ Prediction: {results['predicted_class_name']} (confidence: {confidence:.3f})")
            return results
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """Predict on multiple audio files"""
        results = []
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            results.append(result)
        return results


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Cough Detection Inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--audio_path', help='Path to single audio file')
    parser.add_argument('--audio_dir', help='Directory containing audio files')
    parser.add_argument('--output_file', help='Output file for batch predictions')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create inference object
    try:
        inference = CoughInference(args.model_path, device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize inference: {e}")
        return
    
    # Single file prediction
    if args.audio_path:
        if not os.path.exists(args.audio_path):
            logger.error(f"Audio file not found: {args.audio_path}")
            return
        
        result = inference.predict(args.audio_path)
        
        if 'error' in result:
            logger.error(f"Prediction failed: {result['error']}")
        else:
            print(f"\n🎯 Prediction Results:")
            print(f"   - File: {result['audio_path']}")
            print(f"   - Prediction: {result['predicted_class_name']}")
            print(f"   - Confidence: {result['confidence']:.3f}")
            print(f"   - All probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"     - {class_name}: {prob:.3f}")
    
    # Batch prediction
    elif args.audio_dir:
        if not os.path.exists(args.audio_dir):
            logger.error(f"Audio directory not found: {args.audio_dir}")
            return
        
        # Find audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.webm', '.ogg']:
            audio_files.extend([f for f in os.listdir(args.audio_dir) if f.lower().endswith(ext)])
        
        if not audio_files:
            logger.error(f"No audio files found in {args.audio_dir}")
            return
        
        logger.info(f"📁 Found {len(audio_files)} audio files")
        
        # Predict on all files
        results = []
        for audio_file in audio_files:
            audio_path = os.path.join(args.audio_dir, audio_file)
            result = inference.predict(audio_path)
            results.append(result)
        
        # Save results
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"📁 Results saved to: {args.output_file}")
        
        # Print summary
        successful = len([r for r in results if 'error' not in r])
        print(f"\n📊 Batch Prediction Summary:")
        print(f"   - Total files: {len(audio_files)}")
        print(f"   - Successful: {successful}")
        print(f"   - Failed: {len(audio_files) - successful}")
    
    else:
        logger.error("Please provide either --audio_path or --audio_dir")


if __name__ == "__main__":
    main()
