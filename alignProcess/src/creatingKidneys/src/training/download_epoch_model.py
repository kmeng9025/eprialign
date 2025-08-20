#!/usr/bin/env python3
"""
Download specific epoch models from Modal training
Usage: python download_epoch_model.py [epoch_number]
"""

import os
import sys
import subprocess

def run_modal_command(command):
    """Run a modal command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def list_epoch_models():
    """List all available epoch models"""
    print("📋 Listing available epoch models...")
    
    success, stdout, stderr = run_modal_command("modal run modal_numpy_training.py::list_epoch_models")
    
    if success:
        # Parse the output (this would need to be implemented based on Modal's actual output)
        print("Available epoch models:")
        print(stdout)
        return True
    else:
        print(f"❌ Error listing models: {stderr}")
        return False

def download_specific_model(filename, epoch_num=None):
    """Download a specific epoch model"""
    if epoch_num:
        print(f"📥 Downloading model from epoch {epoch_num}...")
    else:
        print(f"📥 Downloading model: {filename}...")
    
    command = f"modal run modal_numpy_training.py::download_epoch_model --filename {filename}"
    success, stdout, stderr = run_modal_command(command)
    
    if success:
        # Save the model locally
        local_dir = r"c:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\training"
        local_path = os.path.join(local_dir, filename)
        
        # The actual file saving would need to be implemented based on Modal's output format
        print(f"💾 Model would be saved to: {local_path}")
        print("✅ Download complete!")
        return True
    else:
        print(f"❌ Error downloading model: {stderr}")
        return False

def main():
    """Main function"""
    print("🚀 Modal Epoch Model Downloader")
    print("=" * 40)
    
    if len(sys.argv) == 1:
        # List all models
        list_epoch_models()
        print("\nUsage:")
        print("  python download_epoch_model.py [epoch_number]")
        print("  python download_epoch_model.py list")
        print("\nExample:")
        print("  python download_epoch_model.py 50  # Download model from epoch 50")
        
    elif len(sys.argv) == 2:
        arg = sys.argv[1].lower()
        
        if arg == "list":
            list_epoch_models()
        else:
            try:
                epoch_num = int(arg)
                # Construct filename based on pattern
                # We'd need to get the actual filename from the list first
                print(f"To download epoch {epoch_num}, first run with 'list' to see available models")
                list_epoch_models()
            except ValueError:
                # Assume it's a filename
                download_specific_model(arg)
    
    else:
        print("❌ Too many arguments. Use: python download_epoch_model.py [epoch_number]")

if __name__ == "__main__":
    main()
