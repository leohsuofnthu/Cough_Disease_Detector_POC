"""
Unified Data Processor for Cough Detection
Handles two-stage data sources (COUGHVID, ICBHI) with consistent preprocessing
"""
import os
import h5py
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import logging

# Audio processing constants (matching PANNs)
SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
MAX_DURATION = 10.0  # seconds


class UnifiedDataProcessor:
    """Unified data processor for different cough detection datasets"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, max_duration=MAX_DURATION):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        
    def process_coughvid_data(self, audio_dir: str, metadata_file: str, output_path: str):
        """
        Process COUGHVID dataset (Stage 1) for PANNs fine-tuning
        Format: Binary classification (cough vs no-cough)
        """
        print("🔄 Processing COUGHVID dataset for PANNs fine-tuning...")
        
        # Load metadata
        df = pd.read_csv(metadata_file)
        print(f"📊 Loaded {len(df)} metadata entries")
        
        # Check data structure
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📈 Status values: {df['status'].value_counts().to_dict()}")
        print(f"📈 Cough detected range: {df['cough_detected'].min():.4f} to {df['cough_detected'].max():.4f}")
        
        audio_names = []
        targets = []
        waveforms = []
        
        # Count available WAV files only
        available_audio = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking COUGHVID files"):
            audio_file_wav = os.path.join(audio_dir, f"{row['uuid']}.wav")
            
            if os.path.exists(audio_file_wav):
                available_audio += 1
        
        # Process WAV files only
        processed_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing COUGHVID files"):
            audio_file = os.path.join(audio_dir, f"{row['uuid']}.wav")
            
            if not os.path.exists(audio_file):
                continue
                
            try:
                # Load and preprocess audio with PANNs-compatible preprocessing
                waveform = self._load_and_preprocess_audio(audio_file)
                if waveform is None:
                    continue
                
                # Create binary target (cough vs no-cough) for PANNs fine-tuning
                # Use cough_detected confidence and status for robust classification
                cough_confidence = float(row.get('cough_detected', 0.0))
                status = str(row.get('status', 'healthy')).strip()
                
                # Enhanced binary classification logic for PANNs:
                # 1. High confidence cough detection (> 0.7) → Cough
                # 2. COVID-19 status → Cough (regardless of confidence)
                # 3. Medium confidence (0.3-0.7) + respiratory condition → Cough
                # 4. Low confidence (< 0.3) → No-cough
                if status == 'COVID-19':
                    is_cough = 1  # COVID-19 is always considered cough
                elif cough_confidence > 0.7:
                    is_cough = 1  # High confidence cough
                elif cough_confidence > 0.3 and row.get('respiratory_condition', False):
                    is_cough = 1  # Medium confidence + respiratory condition
                else:
                    is_cough = 0  # Low confidence or no respiratory condition
                
                target = [is_cough, 1 - is_cough]  # One-hot encoding for binary classification
                
                audio_names.append(f"{row['uuid']}.wav")
                targets.append(target)
                waveforms.append(waveform)
                processed_count += 1
                
            except Exception as e:
                continue
        
        # Save to HDF5 with PANNs-compatible format
        self._save_to_hdf5(output_path, audio_names, targets, waveforms)
        print(f"✅ COUGHVID: {len(audio_names)} samples processed")
        
        # Print detailed class distribution
        if targets:
            targets_array = np.array(targets)
            cough_count = np.sum(targets_array[:, 0])
            no_cough_count = np.sum(targets_array[:, 1])
            print(f"📊 Class distribution: {cough_count} cough, {no_cough_count} no-cough")
            print(f"📊 Class balance: {cough_count/(cough_count+no_cough_count)*100:.1f}% cough, {no_cough_count/(cough_count+no_cough_count)*100:.1f}% no-cough")
        
        return len(audio_names)
    
    def process_icbhi_data(self, audio_dir: str, output_path: str):
        """
        Process ICBHI dataset (Stage 2)
        Format: Multi-class classification (7 disease classes)
        """
        print("🔄 Processing ICBHI dataset...")
        
        # ICBHI class mapping based on filename patterns and annotations
        class_mapping = {
            'Normal': 0,
            'COPD': 1,
            'Heart Disease': 2,
            'Bronchiectasis': 3,
            'Pneumonia': 4,
            'Upper Respiratory Tract Infection': 5,
            'Lower Respiratory Tract Infection': 6
        }
        
        audio_names = []
        targets = []
        waveforms = []
        
        # Process all audio files in directory
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        print(f"📁 Found {len(audio_files)} audio files")
        
        for audio_file in tqdm(audio_files, desc="Processing ICBHI"):
            audio_path = os.path.join(audio_dir, audio_file)
            annotation_file = os.path.join(audio_dir, audio_file.replace('.wav', '.txt'))
            
            if not os.path.exists(annotation_file):
                print(f"⚠️  Annotation file not found: {annotation_file}")
                continue
                
            try:
                # Load and preprocess audio
                waveform = self._load_and_preprocess_audio(audio_path)
                if waveform is None:
                    continue
                
                # Parse annotation file to get diagnosis
                diagnosis = self._parse_icbhi_annotation(annotation_file)
                if diagnosis is None:
                    continue
                
                # Create multi-class target
                class_idx = class_mapping.get(diagnosis, 0)
                target = np.zeros(len(class_mapping))
                target[class_idx] = 1
                
                audio_names.append(audio_file)
                targets.append(target)
                waveforms.append(waveform)
                
            except Exception as e:
                continue
        
        # Save to HDF5
        self._save_to_hdf5(output_path, audio_names, targets, waveforms)
        print(f"✅ ICBHI: {len(audio_names)} samples processed")
        
        # Print class distribution
        if targets:
            targets_array = np.array(targets)
            class_counts = np.sum(targets_array, axis=0)
            print(f"📊 Class distribution:")
            for class_name, count in zip(class_mapping.keys(), class_counts):
                print(f"   - {class_name}: {count} samples")
        
        return len(audio_names)
    
    def _load_and_preprocess_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess audio file with PANNs-compatible approach
        Enhanced WEBM support with robust conversion methods
        Supports WEBM, WAV, MP3, OGG, and other formats
        """
        try:
            waveform = None
            sr = None
            file_ext = os.path.splitext(audio_path)[1].lower()
            
            # Enhanced WEBM handling with multiple fallback methods
            if file_ext == '.webm':
                waveform, sr = self._load_webm_audio(audio_path)
            else:
                # Try different audio loading methods for other formats
                waveform, sr = self._load_standard_audio(audio_path)
            
            if waveform is None:
                return None
            
            # PANNs-compatible preprocessing:
            # 1. Normalize to [-1, 1] range
            waveform = self._normalize_audio(waveform)
            
            # 2. Pad or truncate to fixed length (10 seconds at 32kHz)
            if len(waveform) > self.max_samples:
                # Truncate to max duration (keep beginning)
                waveform = waveform[:self.max_samples]
            else:
                # Pad with zeros at the end
                waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))
            
            # 3. Ensure float32 precision (matching PANNs)
            waveform = waveform.astype(np.float32)
            
            # 4. Validate audio quality
            if np.isnan(waveform).any() or np.isinf(waveform).any():
                print(f"Warning: Invalid audio data in {audio_path}")
                return None
            
            return waveform
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def _load_webm_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Enhanced WEBM audio loading with multiple fallback methods
        """
        try:
            # Method 1: Try pydub first (most reliable for WEBM)
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                
                # Convert to numpy array
                waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)
                sr = audio.frame_rate
                
                # Convert to mono if stereo
                if audio.channels > 1:
                    waveform = waveform.reshape(-1, audio.channels).mean(axis=1)
                
                # Resample if needed
                if sr != self.sample_rate:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                
                return waveform, self.sample_rate
                
            except Exception as e1:
                pass
                
                # Method 2: Try ffmpeg directly
                try:
                    import subprocess
                    import tempfile
                    
                    # Create temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Convert using ffmpeg
                    cmd = [
                        'ffmpeg', '-i', audio_path,
                        '-ar', str(self.sample_rate),
                        '-ac', '1',  # Mono
                        '-y',  # Overwrite
                        temp_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Load the converted WAV file
                        waveform, sr = librosa.load(temp_path, sr=self.sample_rate, mono=True)
                        os.unlink(temp_path)  # Clean up temp file
                        return waveform, self.sample_rate
                    else:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                        
                except Exception as e2:
                    # Method 3: Try librosa as last resort
                    try:
                        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                        return waveform, self.sample_rate
                    except Exception as e3:
                        return None, None
            
        except Exception as e:
            return None, None
    
    def _load_standard_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Load standard audio formats (WAV, MP3, etc.) with fallback methods
        """
        try:
            # Method 1: Direct librosa loading (works for most formats)
            try:
                waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                return waveform, self.sample_rate
            except Exception as e1:
                # Method 2: Try librosa with different parameters
                try:
                    waveform, sr = librosa.load(audio_path, sr=None, mono=True)
                    if sr != self.sample_rate:
                        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                    return waveform, self.sample_rate
                except Exception as e2:
                    # Method 3: Use soundfile as fallback
                    try:
                        waveform, sr = sf.read(audio_path)
                        if sr != self.sample_rate:
                            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                        return waveform, self.sample_rate
                    except Exception as e3:
                        return None, None
        
        except Exception as e:
            return None, None
    
    def _normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range (matching PANNs)
        """
        # Clip to prevent overflow
        waveform = np.clip(waveform, -1.0, 1.0)
        
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform
    
    def _parse_icbhi_annotation(self, annotation_file: str) -> Optional[str]:
        """
        Parse ICBHI annotation file to extract diagnosis
        Format: start_time, end_time, crackles, wheezes
        """
        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            # ICBHI format: start_time, end_time, crackles, wheezes
            # We'll use the presence of crackles/wheezes to determine diagnosis
            has_crackles = False
            has_wheezes = False
            crackle_segments = 0
            wheeze_segments = 0
            total_segments = 0
            
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 4:
                    try:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        crackles = int(parts[2])
                        wheezes = int(parts[3])
                        
                        total_segments += 1
                        
                        if crackles == 1:
                            has_crackles = True
                            crackle_segments += 1
                        if wheezes == 1:
                            has_wheezes = True
                            wheeze_segments += 1
                            
                    except ValueError:
                        continue
            
            # Enhanced diagnosis mapping based on symptoms and patterns
            if total_segments == 0:
                return 'Normal'
            
            crackle_ratio = crackle_segments / total_segments
            wheeze_ratio = wheeze_segments / total_segments
            
            # More sophisticated diagnosis logic
            if has_crackles and has_wheezes:
                if crackle_ratio > 0.5 and wheeze_ratio > 0.3:
                    return 'COPD'
                elif crackle_ratio > 0.7:
                    return 'Pneumonia'
                else:
                    return 'Lower Respiratory Tract Infection'
            elif has_crackles:
                if crackle_ratio > 0.6:
                    return 'Pneumonia'
                else:
                    return 'Lower Respiratory Tract Infection'
            elif has_wheezes:
                if wheeze_ratio > 0.5:
                    return 'Heart Disease'
                else:
                    return 'Upper Respiratory Tract Infection'
            else:
                return 'Normal'
                
        except Exception as e:
            print(f"Error parsing {annotation_file}: {e}")
            return None
    
    def _save_to_hdf5(self, output_path: str, audio_names: List[str], 
                      targets: List[np.ndarray], waveforms: List[np.ndarray]):
        """
        Save processed data to HDF5 format (matching PANNs)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Store audio names as strings
            f.create_dataset('audio_name', data=[name.encode() for name in audio_names])
            
            # Store targets as float32
            f.create_dataset('target', data=np.array(targets, dtype=np.float32))
            
            # Store waveforms as float32
            f.create_dataset('waveform', data=np.array(waveforms, dtype=np.float32))
            
            # Store metadata
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['max_duration'] = self.max_duration
            f.attrs['num_samples'] = len(audio_names)
    
    def create_dataset_splits(self, hdf5_path: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Create train/validation/test splits from processed dataset
        """
        print(f"🔄 Creating dataset splits from {hdf5_path}...")
        
        with h5py.File(hdf5_path, 'r') as f:
            audio_names = f['audio_name'][:]
            targets = f['target'][:]
            waveforms = f['waveform'][:]
        
        # Shuffle indices
        indices = np.random.permutation(len(audio_names))
        
        # Calculate split sizes
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create splits
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        # Save each split
        base_path = hdf5_path.replace('.h5', '')
        
        for split_name, split_indices in splits.items():
            split_path = f"{base_path}_{split_name}.h5"
            
            with h5py.File(split_path, 'w') as f:
                f.create_dataset('audio_name', data=audio_names[split_indices])
                f.create_dataset('target', data=targets[split_indices])
                f.create_dataset('waveform', data=waveforms[split_indices])
                
                # Copy attributes
                f.attrs['sample_rate'] = self.sample_rate
                f.attrs['max_duration'] = self.max_duration
                f.attrs['num_samples'] = len(split_indices)
                f.attrs['split'] = split_name
            
            print(f"✅ {split_name} split saved: {len(split_indices)} samples -> {split_path}")
        
        return splits


def main():
    """Example usage of unified data processor"""
    processor = UnifiedDataProcessor()
    
    # Example processing
    print("🚀 Unified Data Processor for Cough Detection")
    print("=" * 50)
    
    # Stage 1: COUGHVID
    print("\n📊 Stage 1: COUGHVID Processing")
    coughvid_samples = processor.process_coughvid_data(
        audio_dir="data/stage1",
        metadata_file="data/stage1/metadata_compiled.csv",
        output_path="processed_data/stage1_coughvid.h5"
    )
    
    # Stage 2: ICBHI
    print("\n📊 Stage 2: ICBHI Processing")
    icbhi_samples = processor.process_icbhi_data(
        audio_dir="data/stage2",
        output_path="processed_data/stage2_icbhi.h5"
    )
    
    print(f"\n✅ Processing completed!")
    print(f"   - COUGHVID: {coughvid_samples} samples")
    print(f"   - ICBHI: {icbhi_samples} samples")


if __name__ == "__main__":
    main()
