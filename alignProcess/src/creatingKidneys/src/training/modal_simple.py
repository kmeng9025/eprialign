"""
Simple Modal Kidney Training
============================
Uploads code via volume and runs training on A10 GPU
"""

import modal

app = modal.App("kidney-training-simple")

# Volumes
data_vol = modal.Volume.from_name("kidneyDrawing", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("kidneyCheckpoints", create_if_missing=True)
code_vol = modal.Volume.from_name("kidneyCode", create_if_missing=True)

# Image with dependencies
image = modal.Image.debian_slim().pip_install([
    "torch>=1.10",
    "numpy",
    "scipy", 
    "matplotlib",
])

@app.function(
    image=image,
    gpu="A10G",
    timeout=60*60*8,
    volumes={
        "/data": data_vol,
        "/checkpoints": checkpoints_vol,
        "/code": code_vol,
    }
)
def train_on_modal():
    """Run kidney training on Modal A10 GPU"""
    import subprocess
    import shutil
    import os
    
    print("🚀 Modal AI Kidney Training Started!")
    print("=" * 50)
    
    # Copy code files from volume to working directory
    os.makedirs("/root", exist_ok=True)
    
    for code_file in ["train_fresh_kidney.py", "unet_3d.py"]:
        if os.path.exists(f"/code/{code_file}"):
            shutil.copy(f"/code/{code_file}", f"/root/{code_file}")
            print(f"✅ Copied {code_file}")
        else:
            print(f"❌ Missing {code_file}")
            return f"Missing code file: {code_file}"
    
    # Check training data
    os.makedirs("/data/training", exist_ok=True)
    data_files = [f for f in os.listdir("/data/training") if f.endswith('.mat')]
    print(f"📁 Found {len(data_files)} training files")
    
    if not data_files:
        print("❌ No training data found!")
        return "No training data"
    
    # Change to working directory and run training
    os.chdir("/root")
    
    print("🏋️ Starting training...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen([
        "python3", "train_fresh_kidney.py", "--data_dir", "/data/training"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
       universal_newlines=True, env=env)
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    # Save checkpoints
    print("💾 Saving checkpoints...")
    saved = 0
    for f in os.listdir("/root"):
        if f.endswith(".pth"):
            shutil.copy(f"/root/{f}", f"/checkpoints/{f}")
            print(f"   ✅ Saved {f}")
            saved += 1
    
    return f"Training complete! Saved {saved} checkpoints"

@app.function(volumes={"/code": code_vol})
def upload_code():
    """Upload local code files to volume"""
    import shutil
    import os
    
    print("📤 Uploading code files...")
    
    # Copy the code files that are mounted with the function
    # First, let's see what's available in the current directory
    print("📁 Current directory contents:")
    for f in os.listdir("."):
        print(f"   {f}")
    
    local_files = ["train_fresh_kidney.py", "unet_3d.py"]
    uploaded = 0
    
    for filename in local_files:
        # Look for the file in current directory and parent directories
        found = False
        for path in [".", "..", "../.."]:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                shutil.copy(full_path, f"/code/{filename}")
                print(f"   ✅ Uploaded {filename} from {path}")
                uploaded += 1
                found = True
                break
        
        if not found:
            print(f"   ❌ Not found: {filename}")
    
    return f"Uploaded {uploaded} code files"

@app.function(volumes={"/data": data_vol})
def upload_training_data():
    """Upload local training data"""
    import shutil
    import os
    
    print("📤 Uploading training data...")
    
    # Look for training data
    data_paths = ["../../../data/training", "../../data/training", "training"]
    local_data = None
    
    for path in data_paths:
        if os.path.exists(path):
            local_data = path
            break
    
    if not local_data:
        return "No local training data found"
    
    os.makedirs("/data/training", exist_ok=True)
    
    # Copy .mat files
    uploaded = 0
    for f in os.listdir(local_data):
        if f.endswith('.mat'):
            shutil.copy(f"{local_data}/{f}", f"/data/training/{f}")
            uploaded += 1
    
    print(f"   ✅ Uploaded {uploaded} training files")
    return f"Uploaded {uploaded} files"

if __name__ == "__main__":
    print("🚀 Modal Kidney Training Setup")
    print("=" * 40)
    
    with app.run():
        # Step 1: Upload code
        print("Step 1: Uploading code...")
        code_result = upload_code.remote()
        print(f"Code upload: {code_result}")
        
        # Step 2: Upload data  
        print("\nStep 2: Uploading data...")
        data_result = upload_training_data.remote()
        print(f"Data upload: {data_result}")
        
        # Step 3: Train
        print("\nStep 3: Starting training...")
        training_result = train_on_modal.remote()
        print(f"\n🎉 Final result: {training_result}")
