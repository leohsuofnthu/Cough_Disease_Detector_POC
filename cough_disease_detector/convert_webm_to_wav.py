"""
Simple WEBM to WAV Converter
Converts all WEBM files to WAV files in the same directory
"""
import os
import sys
from pydub import AudioSegment
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_webm_to_wav(directory_path: str, sample_rate: int = 32000):
    """
    Convert all WEBM files to WAV files in the same directory
    
    Args:
        directory_path: Path to directory containing WEBM files
        sample_rate: Target sample rate for WAV files
    """
    print(f"🔄 Converting WEBM files to WAV in: {directory_path}")
    
    # Find all WEBM files
    webm_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith('.webm'):
            webm_files.append(os.path.join(directory_path, file))
    
    if not webm_files:
        print(f"❌ No WEBM files found in {directory_path}")
        return
    
    print(f"📁 Found {len(webm_files)} WEBM files")
    
    successful = 0
    failed = 0
    
    # Convert each WEBM file to WAV
    for webm_file in tqdm(webm_files, desc="Converting WEBM to WAV"):
        try:
            # Create WAV filename (replace .webm with .wav)
            wav_file = webm_file.replace('.webm', '.wav').replace('.WEBM', '.wav')
            
            # Load WEBM file
            audio = AudioSegment.from_file(webm_file)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)
            
            # Export as WAV
            audio.export(wav_file, format="wav")
            
            # Verify file was created
            if os.path.exists(wav_file) and os.path.getsize(wav_file) > 0:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
    
    # Summary
    print(f"\n📊 Conversion Summary:")
    print(f"   - Total files: {len(webm_files)}")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {failed}")
    print(f"   - Success rate: {successful/len(webm_files)*100:.1f}%")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert WEBM files to WAV files in the same directory')
    parser.add_argument('--directory', required=True, help='Directory containing WEBM files')
    parser.add_argument('--sample_rate', type=int, default=32000, help='Target sample rate (default: 32000)')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"❌ Directory not found: {args.directory}")
        sys.exit(1)
    
    # Check if pydub is available
    try:
        from pydub import AudioSegment
        print("✅ Pydub is available")
    except ImportError:
        print("❌ Pydub not found. Install with: pip install pydub")
        print("💡 Also install ffmpeg: conda install ffmpeg")
        sys.exit(1)
    
    # Convert files
    convert_webm_to_wav(args.directory, args.sample_rate)


if __name__ == "__main__":
    main()
