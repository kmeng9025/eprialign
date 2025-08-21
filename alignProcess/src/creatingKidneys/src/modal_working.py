"""
Simple Working Modal Kidney Training
===================================
"""

import modal
import os

app = modal.App("kidney-training-working")

# Create volumes
data_vol = modal.Volume.from_name("kidneyDrawing", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("kidneyCheckpoints", create_if_missing=True)

# Simple image with dependencies
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
    timeout=60*60*8,
    volumes={
        "/data": data_vol,
        "/checkpoints": checkpoints_vol,
    }
)
def run_training():
    """Run kidney training with code uploaded via function"""
    import subprocess
    import shutil
    import os
    
    print("🚀 Modal AI Kidney Training with A10 GPU")
    print("=" * 50)
    
    # Step 1: Create training script in container
    print("📝 Creating training script...")
    
    # Copy the minimal training script content
    train_script = '''#!/usr/bin/env python3
"""Training script for Modal"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from datetime import datetime

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = conv_block(64, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)
        
    def forward(self, x, apply_sigmoid=True):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        logits = self.final(dec1)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        else:
            return logits

class KidneyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.load_data()
    
    def load_data(self):
        print("Loading training data...")
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.mat'):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
                    if 'images' in data:
                        images = data['images']
                        for i in range(len(images)):
                            img = images[i]
                            if hasattr(img, 'Name') and hasattr(img, 'data'):
                                name = img.Name if isinstance(img.Name, str) else 'img'
                                if 'mri' in name.lower() and len(img.data.shape) == 3:
                                    mri_data = img.data
                                    if hasattr(img, 'slaves') and isinstance(img.slaves, np.ndarray):
                                        for slave in img.slaves:
                                            if hasattr(slave, 'Name') and hasattr(slave, 'data'):
                                                slave_name = slave.Name if isinstance(slave.Name, str) else 'slave'
                                                if 'kidney' in slave_name.lower() and 'srf' not in slave_name.lower():
                                                    if len(slave.data.shape) == 3 and slave.data.shape == mri_data.shape:
                                                        self.samples.append({
                                                            'mri': mri_data.astype(np.float32),
                                                            'kidney': (slave.data > 0).astype(np.float32),
                                                            'file': filename,
                                                            'slave_name': slave_name
                                                        })
                                                        print(f"Added sample: {filename} - {slave_name}")
                except Exception as e:
                    print(f"Skipped {filename}: {e}")
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        mri = sample['mri'].copy()
        mask = sample['kidney'].copy()
        p1, p99 = np.percentile(mri, [1, 99])
        mri = (mri - p1) / (p99 - p1)
        mri = np.clip(mri, 0, 1)
        target_shape = (64, 64, 32)
        zoom_factors = [target_shape[i] / mri.shape[i] for i in range(3)]
        mri_resized = zoom(mri, zoom_factors, order=1)
        mask_resized = zoom(mask, zoom_factors, order=0)
        mri_tensor = torch.from_numpy(mri_resized).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
        return mri_tensor, mask_tensor

def train_model():
    print("Starting kidney training on Modal AI Cloud...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = KidneyDataset('/data/training')
    if len(dataset) == 0:
        print("No training data found!")
        return
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    print(f"Starting training with {len(dataset)} samples...")
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for mri, mask in dataloader:
            mri, mask = mri.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(mri)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
        # Save checkpoint
        checkpoint_path = f'/root/kidney_model_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
    
    # Save final model
    final_path = '/root/kidney_unet_model_best.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_path)
    
    print("Training completed!")

if __name__ == "__main__":
    train_model()
'''
    
    with open("/root/train_modal.py", "w") as f:
        f.write(train_script)
    
    print("✅ Training script created")
    
    # Step 2: Check for training data
    print("📁 Checking training data...")
    os.makedirs("/data/training", exist_ok=True)
    data_files = [f for f in os.listdir("/data/training") if f.endswith('.mat')]
    print(f"Found {len(data_files)} .mat files")
    
    if len(data_files) == 0:
        print("❌ No training data found! Please upload .mat files to kidneyDrawing volume")
        return "No training data"
    
    # Step 3: Run training
    print("🏋️ Starting training...")
    os.chdir("/root")
    
    result = subprocess.run([
        "python3", "train_modal.py"
    ], capture_output=False, text=True)
    
    # Step 4: Save checkpoints
    print("💾 Saving checkpoints...")
    saved = 0
    for f in os.listdir("/root"):
        if f.endswith(".pth"):
            shutil.copy(f"/root/{f}", f"/checkpoints/{f}")
            print(f"   ✅ Saved {f}")
            saved += 1
    
    result_msg = f"Training completed! Saved {saved} checkpoints"
    print(f"🎉 {result_msg}")
    return result_msg

@app.function(volumes={"/data": data_vol})
def upload_data():
    """Upload local data to Modal"""
    import shutil
    import os
    
    print("📤 Uploading training data...")
    
    # Look for training data in the correct relative path
    data_paths = [
        "../../../data/training",
        "../../data/training", 
        "../data/training",
        "training",
        "../../../alignProcess/data/training"
    ]
    
    local_data = None
    
    for path in data_paths:
        print(f"   Checking: {path}")
        if os.path.exists(path):
            local_data = path
            print(f"   ✅ Found data in: {path}")
            break
    
    if not local_data:
        print("❌ No local training data found!")
        print("   Checked paths:")
        for path in data_paths:
            print(f"     - {path} (exists: {os.path.exists(path)})")
        return "No local training data found"
    
    os.makedirs("/data/training", exist_ok=True)
    
    # Copy .mat files
    uploaded = 0
    mat_files = [f for f in os.listdir(local_data) if f.endswith('.mat')]
    print(f"   📁 Found {len(mat_files)} .mat files to upload")
    
    for f in mat_files:
        src = os.path.join(local_data, f)
        dst = f"/data/training/{f}"
        shutil.copy2(src, dst)
        file_size = os.path.getsize(src) / (1024*1024)  # MB
        uploaded += 1
        print(f"   ✅ Uploaded: {f} ({file_size:.1f} MB)")
    
    result = f"Uploaded {uploaded} files"
    print(f"📁 {result}")
    return result

if __name__ == "__main__":
    with app.run():
        print("Step 1: Uploading data...")
        upload_result = upload_data.remote()
        print(f"Upload: {upload_result}")
        
        print("Step 2: Running training...")
        training_result = run_training.remote()
        print(f"Training: {training_result}")
