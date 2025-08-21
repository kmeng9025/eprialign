"""
Final Modal Kidney Training - Complete Setup
===========================================
This creates volumes, uploads code and data, then runs training on A10 GPU
"""

import modal
import os
from pathlib import Path

app = modal.App("kidney-training-final")

# Create volumes
data_vol = modal.Volume.from_name("kidneyDrawing", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("kidneyCheckpoints", create_if_missing=True)

# Read local code files
train_code = Path("train_fresh_kidney.py").read_text(encoding='utf-8')
unet_code = Path("unet_3d.py").read_text(encoding='utf-8')

# Create image with dependencies and code files
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=1.10",
        "numpy",
        "scipy", 
        "matplotlib",
    ])
    .run_commands([
        "mkdir -p /root",
        f"cat > /root/train_fresh_kidney.py << 'EOF'\n{train_code}\nEOF",
        f"cat > /root/unet_3d.py << 'EOF'\n{unet_code}\nEOF",
    ])
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60*60*8,
    volumes={
        "/data": data_vol,
        "/checkpoints": checkpoints_vol,
    }
)
def complete_training():
    """Upload data and run complete training pipeline"""
    import subprocess
    import shutil
    import os
    
    print("🚀 Complete Modal AI Kidney Training Pipeline")
    print("=" * 60)
    
    # Step 1: Upload training data if needed
    print("📁 Setting up training data...")
    os.makedirs("/data/training", exist_ok=True)
    
    # Check if we have data, if not try to upload from local
    data_files = [f for f in os.listdir("/data/training") if f.endswith('.mat')]
    print(f"Found {len(data_files)} existing .mat files in volume")
    
    # Step 2: Verify we have the code files
    print("🔧 Checking code files...")
    for code_file in ["train_fresh_kidney.py", "unet_3d.py"]:
        if os.path.exists(f"/root/{code_file}"):
            print(f"   ✅ {code_file} ready")
        else:
            print(f"   ❌ Missing {code_file}")
            return f"Missing code file: {code_file}"
    
    # Step 3: Run training
    print("\n🏋️ Starting kidney detection training...")
    print(f"📊 GPU Available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    os.chdir("/root")
    
    # Run training with real-time output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    cmd = ["python3", "train_fresh_kidney.py", "--data_dir", "/data/training"]
    print(f"🔧 Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, env=env)
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    exit_code = process.returncode
    
    print(f"\n📊 Training completed with exit code: {exit_code}")
    
    # Step 4: Save all model files
    print("💾 Saving model checkpoints...")
    saved_files = []
    
    for filename in os.listdir("/root"):
        if filename.endswith(".pth") and ("kidney" in filename.lower() or "model" in filename.lower()):
            src_path = f"/root/{filename}"
            dst_path = f"/checkpoints/{filename}"
            shutil.copy2(src_path, dst_path)
            file_size = os.path.getsize(src_path) / (1024*1024)  # MB
            saved_files.append(filename)
            print(f"   ✅ Saved: {filename} ({file_size:.1f} MB)")
    
    if not saved_files:
        print("   ⚠️  No model files found to save!")
        # List all files for debugging
        print("   Debug - all files in /root:")
        for f in os.listdir("/root"):
            print(f"      {f}")
    
    result = f"Training completed! Exit code: {exit_code}, Saved {len(saved_files)} model files"
    print(f"\n🎉 {result}")
    return result

@app.function(volumes={"/data": data_vol})
def upload_local_data():
    """Upload any local training data found"""
    import shutil
    import os
    
    print("📤 Looking for local training data to upload...")
    
    # Look for training data in common locations
    data_paths = [
        "../../../data/training",
        "../../data/training", 
        "../training",
        "training"
    ]
    
    local_data_dir = None
    for path in data_paths:
        if os.path.exists(path):
            local_data_dir = path
            print(f"   Found data in: {path}")
            break
    
    if not local_data_dir:
        print("   ❌ No local training data found")
        return "No local data found"
    
    # Create remote directory
    os.makedirs("/data/training", exist_ok=True)
    
    # Copy .mat files
    mat_files = [f for f in os.listdir(local_data_dir) if f.endswith('.mat')]
    print(f"   📁 Uploading {len(mat_files)} .mat files...")
    
    for mat_file in mat_files:
        src = os.path.join(local_data_dir, mat_file)
        dst = f"/data/training/{mat_file}"
        shutil.copy2(src, dst)
        file_size = os.path.getsize(src) / (1024*1024)  # MB
        print(f"   ✅ Uploaded: {mat_file} ({file_size:.1f} MB)")
    
    result = f"Uploaded {len(mat_files)} training files"
    print(f"📁 {result}")
    return result

if __name__ == "__main__":
    print("🚀 Modal AI Kidney Training - Complete Setup")
    print("=" * 60)
    
    with app.run():
        # Upload training data first
        print("Step 1: Uploading training data...")
        try:
            upload_result = upload_local_data.remote()
            print(f"Upload result: {upload_result}")
        except Exception as e:
            print(f"Data upload failed (may already exist): {e}")
        
        # Run complete training pipeline
        print("\nStep 2: Running complete training...")
        training_result = complete_training.remote()
        print(f"\n🎉 FINAL RESULT: {training_result}")
        
        print("\n✅ All done! Check your kidneyCheckpoints volume for the trained model.")
