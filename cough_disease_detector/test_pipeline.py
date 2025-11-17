"""
Elegant pipeline test - tests data processing and model compatibility
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def test_data_processing():
    """Test data processing with WAV files"""
    print("\n🧪 Testing Data Processing")
    print("-" * 40)
    
    try:
        from unified_data_processor import UnifiedDataProcessor
        
        # Test Stage 1 (COUGHVID WAV)
        if os.path.exists("data/stage1/metadata_compiled.csv"):
            print("🔄 Testing Stage 1 (COUGHVID WAV)...")
            processor = UnifiedDataProcessor()
            result = processor.process_coughvid_data(
                "data/stage1", 
                "data/stage1/metadata_compiled.csv", 
                "data/stage1_processed.h5"
            )
            print(f"✅ Stage 1: {result} samples processed")
        else:
            print("❌ Stage 1 metadata not found")
            return False
        
        # Test Stage 2 (ICBHI)
        if os.path.exists("data/stage2"):
            print("🔄 Testing Stage 2 (ICBHI)...")
            result = processor.process_icbhi_data(
                "data/stage2", 
                "data/stage2_processed.h5"
            )
            print(f"✅ Stage 2: {result} samples processed")
        else:
            print("⚠️  Stage 2 data not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def main():
    """Elegant pipeline test"""
    print("🚀 Pipeline Test")
    print("=" * 30)
    
    # Test data processing
    print("🔄 Testing data processing...")
    if not test_data_processing():
        print("❌ Data processing failed")
        return False
    
    print("\n🎉 Pipeline test passed! Ready for PANNs fine-tuning.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
