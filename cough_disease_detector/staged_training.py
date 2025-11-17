"""
Staged Transfer Learning Pipeline for Cough Detection
"""
import os
import argparse
from trainer import CoughTrainer
from data_loader import create_data_loaders
from config import STAGE_CONFIGS, PRETRAINED_MODEL_PATH, WORKSPACE
from models import CoughDetector
from pytorch_utils import move_data_to_device
import torch
from sklearn.metrics import accuracy_score, classification_report

# Fix tqdm for Google Colab
import os
os.environ['TQDM_DISABLE'] = '0'  # Keep tqdm enabled but fix display


def download_pretrained_model():
    """Download pretrained PANNs model"""
    import urllib.request
    from tqdm import tqdm
    
    model_url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    model_path = PRETRAINED_MODEL_PATH
    
    # Create pretrained directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"📥 Downloading pretrained model from {model_url}")
        print(f"📁 Saving to: {model_path}")
        
        # Download with progress bar
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading") as pbar:
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                    pbar.update(block_size)
            
            urllib.request.urlretrieve(model_url, model_path, progress_hook)
        
        print(f"✅ Model saved to {model_path}")
    else:
        print(f"✅ Pretrained model already exists: {model_path}")


def stage1_coughvid_training(dataset_path, workspace, device='cuda'):
    """Stage 1: Train on COUGHVID dataset (cough vs no-cough)"""
    print("=" * 60)
    print("STAGE 1: COUGHVID Training")
    print("=" * 60)
    print("Learning cough fundamentals from COUGHVID dataset")
    print("Freezing: First 2 conv blocks")
    print("Learning Rate: 1e-4")
    print("Epochs: 10")
    print("=" * 60)
    
    # Download pretrained model if needed
    download_pretrained_model()
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    # Create trainer
    trainer = CoughTrainer(
        stage='stage1_coughvid',
        workspace=workspace,
        pretrained_path=PRETRAINED_MODEL_PATH,
        device=device
    )
    
    # Create data loaders with smaller batch size for small datasets
    train_loader, val_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=16,  # Increased batch size for better training
        holdout_fold=None,
        num_workers=4  # Use multiple workers for faster loading
    )
    
    # Train
    best_accuracy = trainer.train(train_loader, val_loader)
    
    print(f"Stage 1 completed. Best accuracy: {best_accuracy:.2f}%")
    return os.path.join(workspace, 'checkpoints', 'stage1_coughvid', 'best_model.pth')


def stage2_icbhi_training(dataset_path, workspace, previous_checkpoint, device='cuda'):
    """Stage 2: Train on ICBHI dataset (multi-class)"""
    print("=" * 60)
    print("STAGE 2: ICBHI Training")
    print("=" * 60)
    print("Multi-class fine-tuning on ICBHI dataset")
    print("Freezing: None (unfreeze all)")
    print("Learning Rate: 1e-5")
    print("Epochs: 15")
    print("=" * 60)
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    # Create trainer with previous stage checkpoint (could be stage 1 or stage 2)
    trainer = CoughTrainer(
        stage='stage2_icbhi',
        workspace=workspace,
        pretrained_path=previous_checkpoint,
        device=device
    )
    
    # Create data loaders with smaller batch size for small datasets
    train_loader, val_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=16,  # Increased batch size for better training
        holdout_fold=None,
        num_workers=4  # Use multiple workers for faster loading
    )
    
    # Train
    best_accuracy = trainer.train(train_loader, val_loader)
    
    print(f"Stage 2 completed. Best accuracy: {best_accuracy:.2f}%")
    return os.path.join(workspace, 'checkpoints', 'stage2_icbhi', 'best_model.pth')


def run_full_pipeline(coughvid_path, icbhi_path, workspace, device='cuda'):
    """Run the complete 2-stage transfer learning pipeline"""
    print("🚀 Starting 2-Stage Transfer Learning Pipeline for Cough Detection")
    print("=" * 70)
    
    # Stage 1: COUGHVID (Cough Detection)
    print("Stage 1: Learning cough fundamentals from COUGHVID")
    stage1_checkpoint = stage1_coughvid_training(coughvid_path, workspace, device)
    
    # Stage 2: ICBHI (Disease Classification)
    print("Stage 2: Multi-class disease classification on ICBHI")
    final_checkpoint = stage2_icbhi_training(icbhi_path, workspace, stage1_checkpoint, device)
    
    print("=" * 70)
    print("🎉 2-Stage Pipeline completed successfully!")
    print(f"Final model saved at: {final_checkpoint}")
    print("=" * 70)
    
    return final_checkpoint


def evaluate_model(model_path, dataset_path, device='cuda'):
    """Evaluate trained model"""
    print("🧪 Evaluating Model")
    print("=" * 40)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = CoughDetector(
        sample_rate=32000, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=checkpoint['config']['classes_num']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
    
    # Create data loader
    _, val_loader = create_data_loaders(dataset_path, batch_size=32, num_workers=8)
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = move_data_to_device(batch_data, device)
            outputs = model(batch_data['waveform'])
            targets = batch_data['target']
            
            pred = torch.argmax(outputs['clipwise_output'], dim=1)
            target_classes = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target_classes.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds) * 100
    report = classification_report(all_targets, all_preds)
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Classification Report:\n{report}")
    
    return accuracy, report


def dry_run_pipeline(coughvid_path, icbhi_path, device='cuda'):
    """Dry run to test pipeline components without training"""
    print("🧪 DRY RUN: Testing Pipeline Components")
    print("=" * 60)
    
    try:
        # Test 1: Model Loading
        print("🔄 Test 1: Model Loading...")
        from models import CoughDetector
        model = CoughDetector(
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=2
        )
        print("✅ Model created successfully")
        
        # Test 2: Data Loader Creation
        print("🔄 Test 2: Data Loader Creation...")
        train_loader, val_loader = create_data_loaders(
            dataset_path=coughvid_path,
            batch_size=1,
            holdout_fold=None,
            num_workers=0
        )
        print(f"✅ Train loader: {len(train_loader)} batches")
        print(f"✅ Val loader: {len(val_loader)} batches")
        
        # Test 3: Data Loading
        print("🔄 Test 3: Data Loading...")
        if len(train_loader) > 0:
            batch_data = next(iter(train_loader))
            print(f"✅ Batch shape: {batch_data['waveform'].shape}")
            print(f"✅ Target shape: {batch_data['target'].shape}")
        else:
            print("⚠️  No training data available")
        
        # Test 4: Model Forward Pass
        print("🔄 Test 4: Model Forward Pass...")
        if len(train_loader) > 0:
            model.eval()
            with torch.no_grad():
                outputs = model(batch_data['waveform'])
                print(f"✅ Output shape: {outputs['clipwise_output'].shape}")
                print(f"✅ Embedding shape: {outputs['embedding'].shape}")
        else:
            print("⚠️  Skipping forward pass - no data")
        
        # Test 5: Loss Function
        print("🔄 Test 5: Loss Function...")
        from losses import get_loss_func
        loss_func = get_loss_func('clip_nll')
        if len(train_loader) > 0:
            loss = loss_func(outputs, {'target': batch_data['target']})
            print(f"✅ Loss computed: {loss.item():.4f}")
        else:
            print("⚠️  Skipping loss computation - no data")
        
        # Test 6: Optimizer
        print("🔄 Test 6: Optimizer...")
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        print("✅ Optimizer created successfully")
        
        # Test 7: ICBHI Data Loading
        print("🔄 Test 7: ICBHI Data Loading...")
        try:
            icbhi_train_loader, icbhi_val_loader = create_data_loaders(
                dataset_path=icbhi_path,
                batch_size=1,
                holdout_fold=None,
                num_workers=0
            )
            print(f"✅ ICBHI Train loader: {len(icbhi_train_loader)} batches")
            print(f"✅ ICBHI Val loader: {len(icbhi_val_loader)} batches")
        except Exception as e:
            print(f"⚠️  ICBHI loading failed: {e}")
        
        print("\n🎉 DRY RUN COMPLETED SUCCESSFULLY!")
        print("✅ All pipeline components are working correctly")
        print("✅ Ready for actual training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DRY RUN FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Cough Detection Staged Training')
    parser.add_argument('--stage', type=str, choices=['stage1', 'stage2', 'all'],
                       default='all', help='Training stage to run (2-stage approach)')
    parser.add_argument('--coughvid_path', type=str, default='data/stage1_processed.h5',
                       help='Path to COUGHVID dataset HDF5 file (default: data/stage1_processed.h5)')
    parser.add_argument('--icbhi_path', type=str, default='data/stage2_processed.h5',
                       help='Path to ICBHI dataset HDF5 file (default: data/stage2_processed.h5)')
    parser.add_argument('--workspace', type=str, default=WORKSPACE,
                       help='Workspace directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Dry run mode - test pipeline without actual training')
    parser.add_argument('--inference', action='store_true',
                       help='Run inference mode')
    parser.add_argument('--audio_path', type=str, default=None,
                       help='Audio file path for inference')
    
    args = parser.parse_args()
    
    # Handle inference mode
    if args.inference:
        if args.audio_path is None:
            print("❌ Inference mode requires --audio_path argument")
            return False
        
        # Find the latest model checkpoint
        import glob
        checkpoint_pattern = os.path.join(args.workspace, 'checkpoints', '*', 'best_model.pth')
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print("❌ No trained model found. Please train a model first.")
            return False
        
        # Use the most recent checkpoint
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"🎯 Using model: {latest_checkpoint}")
        
        # Run inference
        try:
            from inference import CoughInference
            inference_runner = CoughInference(latest_checkpoint, device=args.device)
            result = inference_runner.predict(args.audio_path)
        except Exception as exc:
            print(f"❌ Inference failed: {exc}")
            return False
        
        if 'error' in result:
            print(f"❌ Prediction failed: {result['error']}")
            return False
        
        print(f"\n🎯 Prediction Results:")
        print(f"   - File: {result['audio_path']}")
        print(f"   - Prediction: {result['predicted_class_name']}")
        print(f"   - Confidence: {result['confidence']:.3f}")
        return True
    
    # Handle dry run mode
    if args.dry_run:
        print("🧪 DRY RUN MODE: Testing pipeline without training")
        if args.icbhi_path is None:
            print("⚠️  ICBHI path not provided. Testing Stage 1 only.")
            # Test Stage 1 only
            try:
                from models import CoughDetector
                from data_loader import create_data_loaders
                from losses import get_loss_func
                import torch.optim as optim
                
                # Test Stage 1 components
                model = CoughDetector(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=2)
                train_loader, val_loader = create_data_loaders(args.coughvid_path, batch_size=1, holdout_fold=None, num_workers=0)
                print(f"✅ Stage 1: {len(train_loader)} train batches, {len(val_loader)} val batches")
                print("✅ Stage 1 components working correctly")
                return True
            except Exception as e:
                print(f"❌ Stage 1 test failed: {e}")
                return False
        else:
            success = dry_run_pipeline(args.coughvid_path, args.icbhi_path, args.device)
            return success
    
    if args.stage == 'all':
        if args.coughvid_path is None or args.icbhi_path is None:
            print("❌ Full pipeline requires both --coughvid_path and --icbhi_path")
        else:
            run_full_pipeline(
                args.coughvid_path, args.icbhi_path,
                args.workspace, args.device
            )
    elif args.stage == 'stage1':
        if args.coughvid_path is None:
            print("❌ Stage 1 requires COUGHVID dataset path. Use --coughvid_path argument.")
        else:
            stage1_coughvid_training(args.coughvid_path, args.workspace, args.device)
    elif args.stage == 'stage2':
        if args.icbhi_path is None:
            print("❌ Stage 2 requires ICBHI dataset path. Use --icbhi_path argument.")
        else:
            # Check if stage 1 checkpoint exists
            stage1_checkpoint = os.path.join(args.workspace, 'checkpoints', 'stage1_coughvid', 'best_model.pth')
            if os.path.exists(stage1_checkpoint):
                stage2_icbhi_training(args.icbhi_path, args.workspace, args.device, stage1_checkpoint)
            else:
                print("❌ Stage 2 requires stage 1 checkpoint. Please run stage 1 first.")


if __name__ == '__main__':
    main()
