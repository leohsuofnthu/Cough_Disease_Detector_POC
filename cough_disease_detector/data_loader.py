"""
Enhanced Data loader for cough detection datasets
Unified approach matching PANNs preprocessing
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple

class CoughDataset(Dataset):
    """Enhanced cough detection dataset with PANNs-compatible preprocessing"""
    def __init__(self, hdf5_path: str, sample_rate: int = 32000, 
                 max_duration: float = 10.0, augment: bool = False):
        self.hdf5_path = hdf5_path
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.augment = augment
        
        # Load dataset info
        with h5py.File(hdf5_path, 'r') as f:
            self.audio_names = f['audio_name'][:]
            self.targets = f['target'][:]
            self.fold = f['fold'][:] if 'fold' in f.keys() else None
            
            # Load metadata if available
            self.sample_rate_orig = f.attrs.get('sample_rate', sample_rate)
            self.max_duration_orig = f.attrs.get('max_duration', max_duration)
    
    def __len__(self):
        return len(self.audio_names)
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            audio_name = f['audio_name'][idx].decode()
            target = f['target'][idx]
            waveform = f['waveform'][idx]
            
            # Ensure waveform is float32
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            
            # Pad or truncate to fixed length
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            else:
                waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))
            
            # Apply augmentation if training
            if self.augment and self.training:
                waveform = self._apply_augmentation(waveform)
            
            return {
                'audio_name': audio_name,
                'waveform': torch.FloatTensor(waveform),
                'target': torch.FloatTensor(target)
            }
    
    def _apply_augmentation(self, waveform: np.ndarray) -> np.ndarray:
        """Apply audio augmentation (matching PANNs approach)"""
        # Time stretching
        if np.random.random() < 0.5:
            stretch_factor = np.random.uniform(0.8, 1.2)
            waveform = librosa.effects.time_stretch(waveform, rate=stretch_factor)
        
        # Pitch shifting
        if np.random.random() < 0.5:
            pitch_shift = np.random.randint(-2, 3)
            waveform = librosa.effects.pitch_shift(waveform, sr=self.sample_rate, n_steps=pitch_shift)
        
        # Add noise
        if np.random.random() < 0.3:
            noise_factor = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_factor, len(waveform))
            waveform = waveform + noise
        
        return waveform


class BalancedTrainSampler:
    """Balanced training sampler (matching PANNs approach)"""
    def __init__(self, hdf5_path: str, batch_size: int, holdout_fold: Optional[int] = None):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.holdout_fold = holdout_fold
        
        # Load dataset info
        with h5py.File(hdf5_path, 'r') as f:
            self.audio_names = f['audio_name'][:]
            self.targets = f['target'][:]
            self.fold = f['fold'][:] if 'fold' in f.keys() else None
        
        # Get training indices
        if holdout_fold is not None and self.fold is not None:
            self.indexes = np.where(self.fold != holdout_fold)[0]
        else:
            self.indexes = np.arange(len(self.audio_names))
        
        # Balance classes
        self._balance_classes()
        
        self.n_samples = len(self.indexes)
        self.n_batches = self.n_samples // self.batch_size
    
    def _balance_classes(self):
        """Balance classes for training"""
        # Get class labels
        if len(self.targets) == 0:
            self.indexes = np.array([])
            return
            
        class_labels = np.argmax(self.targets[self.indexes], axis=1)
        
        # Count samples per class
        unique_classes, counts = np.unique(class_labels, return_counts=True)
        max_count = np.max(counts)
        
        # Oversample minority classes
        balanced_indexes = []
        for class_id in unique_classes:
            class_indices = self.indexes[class_labels == class_id]
            
            # Oversample if needed
            if len(class_indices) < max_count:
                oversample_count = max_count - len(class_indices)
                oversample_indices = np.random.choice(
                    class_indices, size=oversample_count, replace=True
                )
                balanced_indexes.extend(class_indices)
                balanced_indexes.extend(oversample_indices)
            else:
                balanced_indexes.extend(class_indices)
        
        self.indexes = np.array(balanced_indexes)
    
    def __iter__(self):
        # Shuffle indexes
        np.random.shuffle(self.indexes)
        
        for i in range(self.n_batches):
            batch_indexes = self.indexes[i * self.batch_size:(i + 1) * self.batch_size]
            yield batch_indexes
    
    def __len__(self):
        return self.n_batches


class EvaluateSampler:
    """Evaluation sampler"""
    def __init__(self, hdf5_path, batch_size, holdout_fold=None):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.holdout_fold = holdout_fold
        
        # Get evaluation indices
        with h5py.File(hdf5_path, 'r') as f:
            if holdout_fold is not None and 'fold' in f.keys():
                folds = f['fold'][:]
                self.indexes = np.where(folds == holdout_fold)[0]
            else:
                # Use last 20% for evaluation
                n_samples = len(f['audio_name'])
                self.indexes = np.arange(int(0.8 * n_samples), n_samples)
        
        self.n_samples = len(self.indexes)
        self.n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_samples)
            batch_indexes = self.indexes[start_idx:end_idx]
            yield batch_indexes
    
    def __len__(self):
        return self.n_batches


def collate_fn(batch):
    """Collate function for DataLoader"""
    audio_names = [item['audio_name'] for item in batch]
    waveforms = torch.stack([item['waveform'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    return {
        'audio_name': audio_names,
        'waveform': waveforms,
        'target': targets
    }


def create_data_loaders(dataset_path: str, batch_size: int, 
                        holdout_fold: Optional[int] = None, 
                        num_workers: int = 8, augment: bool = False):
    """Create enhanced training and validation data loaders"""
    dataset = CoughDataset(dataset_path, augment=augment)
    
    train_sampler = BalancedTrainSampler(dataset_path, batch_size, holdout_fold)
    val_sampler = EvaluateSampler(dataset_path, batch_size, holdout_fold)
    
    # Disable pin_memory if no GPU available
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        dataset=dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

