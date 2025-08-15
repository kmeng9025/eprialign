import modal
import os
from pathlib import Path

# Create Modal volumes (will create if they don't exist)
KIDNEY_DATA_VOL = modal.Volume.from_name("kidneyDrawing", create_if_missing=True)
KIDNEY_CHECKPOINTS_VOL = modal.Volume.from_name("kidneyCheckpoints", create_if_missing=True)

# Modal image with PyTorch and dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=1.10",
        "numpy",
        "scipy", 
        "matplotlib"
    ])
    .run_commands(["apt-get update && apt-get install -y python3-tk"])
)

app = modal.App("kidney-train-app")

@app.function(
    image=image,
    gpu="A10G",
    timeout=60*60*6,  # 6 hours
    volumes={
        "/mnt/data": KIDNEY_DATA_VOL,
        "/mnt/checkpoints": KIDNEY_CHECKPOINTS_VOL,
    },
)
def train_kidney_model():
    """Upload code, run training, and save results"""
    import subprocess
    import shutil
    
    print("🚀 Starting Modal AI kidney training...")
    
    # Change to working directory
    os.chdir("/root")
    
    # Run the training script
    print("🏋️ Running training script...")
    result = subprocess.run([
        "python3", "train_fresh_kidney.py", "--data_dir", "/mnt/data/training"
    ], capture_output=False, text=True)
    
    print(f"Training completed with exit code: {result.returncode}")
    
    # Move all model files to checkpoints volume
    print("💾 Saving checkpoints...")
    for f in os.listdir("/root"):
        if f.endswith(".pth") and ("kidney" in f or "model" in f):
            src = f"/root/{f}"
            dst = f"/mnt/checkpoints/{f}"
            shutil.copy2(src, dst)
            print(f"   Saved: {f}")
    
    print("✅ Training and checkpoint saving complete!")
    return "Training completed successfully"

@app.function(volumes={"/mnt/data": KIDNEY_DATA_VOL})
def upload_training_data():
    """Upload local training data to Modal volume"""
    import shutil
    
    print("📁 Creating training directory structure...")
    os.makedirs("/mnt/data/training", exist_ok=True)
    
    # Files will be uploaded by Modal's built-in mechanisms
    print("✅ Data directory ready for upload")

if __name__ == "__main__":
    # First, make sure we have the volumes set up
    print("🔧 Setting up Modal volumes...")
    
    # Upload code files to the app
    local_files = {
        "train_fresh_kidney.py": Path("train_fresh_kidney.py"),
        "unet_3d.py": Path("unet_3d.py"),
        "requirements.txt": Path("requirements.txt")
    }
    
    # Copy files to the image at build time
    for remote_path, local_path in local_files.items():
        if local_path.exists():
            image = image.copy_local_file(str(local_path), f"/root/{remote_path}")
    
    # Run the training
    with app.run():
        print("🚀 Launching training job...")
        result = train_kidney_model.remote()
        print(f"Result: {result}")
