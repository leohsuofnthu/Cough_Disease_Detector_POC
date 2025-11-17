"""
Training module for cough detection with staged transfer learning
"""
import os
import time
import logging
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Fix tqdm for Google Colab
try:
    from tqdm.auto import tqdm
    # Configure tqdm for better Colab display
    import tqdm as tqdm_module
    tqdm_module.tqdm.monitor_interval = 0
except ImportError:
    from tqdm import tqdm

from models import CoughDetector
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from data_loader import create_data_loaders
from config import STAGE_CONFIGS, LOG_INTERVAL, SAVE_INTERVAL


class CoughTrainer:
    """Cough Detection Trainer with staged transfer learning"""
    
    def __init__(self, stage, workspace, pretrained_path=None, device='cuda'):
        self.stage = stage
        self.workspace = workspace
        self.device = device
        self.config = STAGE_CONFIGS[stage]
        
        # Create directories
        self.checkpoint_dir = os.path.join(workspace, 'checkpoints', stage)
        self.log_dir = os.path.join(workspace, 'logs', stage)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize model
        self.model = CoughDetector(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=self.config['classes_num'],
            freeze_blocks=self.config['freeze_blocks'],
            pretrained_path=pretrained_path
        )
        
        if torch.cuda.is_available() and device == 'cuda':
            self.model = self.model.cuda()
        else:
            if device == 'cuda':
                print("Warning: CUDA not available, using CPU")
                self.device = 'cpu'
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.,
            amsgrad=True
        )
        
        # Loss function
        self.loss_func = get_loss_func('clip_nll')
        
        self.logger.info(f"Initialized {stage} trainer")
        self.logger.info(f"Config: {self.config}")
    
    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger(self.stage)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(
                os.path.join(self.log_dir, f'{self.stage}.log')
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Handle empty data loader
        if len(train_loader) == 0:
            self.logger.warning(f'Epoch {epoch}: No training data available')
            return 0.0, 0.0
        
        # Create progress bar with Colab-friendly settings
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', 
                   leave=False, ncols=100, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   position=0, dynamic_ncols=True)
        
        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            batch_data = move_data_to_device(batch_data, self.device)
            
            # Forward pass
            outputs = self.model(batch_data['waveform'])
            targets = batch_data['target']
            
            # Calculate loss
            loss = self.loss_func(outputs, {'target': targets})
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = torch.argmax(outputs['clipwise_output'], dim=1)
            target_classes = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets
            correct += (pred == target_classes).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total if total > 0 else 0.0
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
            if batch_idx % LOG_INTERVAL == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%'
                )
        
        # Handle division by zero
        if len(train_loader) == 0:
            avg_loss = 0.0
            accuracy = 0.0
        else:
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Evaluate model with progress bar"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        # Handle empty data loader
        if len(val_loader) == 0:
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'predictions': [],
                'targets': []
            }
        
        # Create progress bar for evaluation with Colab-friendly settings
        pbar = tqdm(val_loader, desc='[Eval]', 
                   leave=False, ncols=100, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   position=0, dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_data in pbar:
                batch_data = move_data_to_device(batch_data, self.device)
                
                outputs = self.model(batch_data['waveform'])
                targets = batch_data['target']
                
                loss = self.loss_func(outputs, {'target': targets})
                total_loss += loss.item()
                
                pred = torch.argmax(outputs['clipwise_output'], dim=1)
                target_classes = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target_classes.cpu().numpy())
                
                # Update progress bar
                current_loss = total_loss / len(all_preds) if len(all_preds) > 0 else 0.0
                pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # Handle division by zero
        if len(val_loader) == 0:
            avg_loss = 0.0
            accuracy = 0.0
        else:
            avg_loss = total_loss / len(val_loader)
            accuracy = accuracy_score(all_targets, all_preds) * 100 if len(all_targets) > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def train(self, train_loader, val_loader, resume_epoch=0):
        """Full training loop with enhanced monitoring"""
        best_accuracy = 0
        
        print(f"\nStarting Training: {self.stage}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print(f"Freeze Blocks: {self.config['freeze_blocks']}")
        print("=" * 60)
        
        for epoch in range(resume_epoch, self.config['epochs']):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            
            epoch_time = time.time() - start_time
            
            # Enhanced epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"  Time:  {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  New best model! Accuracy: {val_acc:.2f}%")
            else:
                print(f"  Best so far: {best_accuracy:.2f}%")
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
                print(f"  Checkpoint saved")
        
        print(f"\nTraining completed!")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        print("=" * 60)
        
        self.logger.info(f'Training completed. Best accuracy: {best_accuracy:.2f}%')
        return best_accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        self.logger.info(f'Checkpoint saved: {path}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        self.logger.info(f'Checkpoint loaded: {checkpoint_path}')
        return epoch
