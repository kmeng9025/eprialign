"""
Modal AI Cloud Kidney Training Script
=====================================
This script uploads your code and data to Modal AI Cloud and runs the training.
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("kidney-training")

# Create volumes
data_volume = modal.Volume.from_name("kidneyDrawing", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("kidneyCheckpoints", create_if_missing=True)

# Create image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=1.10",
        "numpy", 
        "scipy",
        "matplotlib",
    ])
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60*60*8,  # 8 hours timeout
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    mounts=[
        modal.Mount.from_local_file("train_fresh_kidney.py", remote_path="/root/train_fresh_kidney.py"),
        modal.Mount.from_local_file("unet_3d.py", remote_path="/root/unet_3d.py"),
    ]
)
def train_kidney_model():
    """Run the kidney training on Modal with A10 GPU"""
    import subprocess
    import shutil
    
    print("🚀 Starting kidney training on Modal AI Cloud (A10 GPU)...")
    print("=" * 60)
    
    # Create training data directory if it doesn't exist
    os.makedirs("/data/training", exist_ok=True)
    
    # Check if we have training data
    if not os.path.exists("/data/training") or not os.listdir("/data/training"):
        print("⚠️  No training data found in /data/training")
        print("Please upload your .mat files to the kidneyDrawing volume first")
        return "No training data found"
    
    # List available data
    data_files = [f for f in os.listdir("/data/training") if f.endswith('.mat')]
    print(f"📁 Found {len(data_files)} .mat files for training:")
    for f in data_files[:5]:  # Show first 5
        print(f"   - {f}")
    if len(data_files) > 5:
        print(f"   ... and {len(data_files) - 5} more")
    
    # Change to working directory
    os.chdir("/root")
    
    # Run the training script
    print("\n🏋️ Starting training...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen([
        "python3", "train_fresh_kidney.py", "--data_dir", "/data/training"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
       universal_newlines=True, env=env)
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    exit_code = process.returncode
    
    print(f"\n📊 Training completed with exit code: {exit_code}")
    
    # Save all model files to checkpoint volume
    print("💾 Saving model checkpoints...")
    saved_files = []
    
    for filename in os.listdir("/root"):
        if filename.endswith(".pth") and ("kidney" in filename.lower() or "model" in filename.lower()):
            src_path = f"/root/{filename}"
            dst_path = f"/checkpoints/{filename}"
            shutil.copy2(src_path, dst_path)
            saved_files.append(filename)
            print(f"   ✅ Saved: {filename}")
    
    if not saved_files:
        print("   ⚠️  No model files found to save")
    
    print(f"\n✅ Training complete! Saved {len(saved_files)} model files to kidneyCheckpoints volume")
    return f"Training completed successfully. Saved {len(saved_files)} model files."

@app.function(volumes={"/data": data_volume})
def upload_local_data():
    """Upload local training data to Modal volume"""
    import shutil
    
    print("📤 Uploading local training data...")
    
    # Look for local training data
    local_data_paths = [
        "../../../data/training",
        "../../data/training", 
        "training",
        "../training"
    ]
    
    local_data_dir = None
    for path in local_data_paths:
        if os.path.exists(path):
            local_data_dir = path
            break
    
    if not local_data_dir:
        print("❌ No local training data found")
        return "No local data found"
    
    # Create remote directory
    os.makedirs("/data/training", exist_ok=True)
    
    # Copy .mat files
    mat_files = [f for f in os.listdir(local_data_dir) if f.endswith('.mat')]
    
    for mat_file in mat_files:
        src = os.path.join(local_data_dir, mat_file)
        dst = f"/data/training/{mat_file}"
        shutil.copy2(src, dst)
        print(f"   ✅ Uploaded: {mat_file}")
    
    print(f"📁 Uploaded {len(mat_files)} training files")
    return f"Uploaded {len(mat_files)} files"

if __name__ == "__main__":
    print("🚀 Modal AI Cloud Kidney Training Setup")
    print("=" * 50)
    
    with app.run():
        # First upload data (if available locally)
        print("Step 1: Uploading training data...")
        try:
            upload_result = upload_local_data.remote()
            print(f"Upload result: {upload_result}")
        except Exception as e:
            print(f"Upload failed (this is OK if data is already uploaded): {e}")
        
        # Then run training
        print("\nStep 2: Starting training...")
        training_result = train_kidney_model.remote()
        print(f"\nFinal result: {training_result}")
