#!/usr/bin/env python3
"""
Modal Training with Numpy Data Upload
====================================

Upload numpy training data and train kidney detection model on Modal A10 GPU.

Author: AI Assistant
Date: 2025-08-15
"""

import modal
import os
import numpy as np
import pickle
from pathlib import Path

# Modal setup
app = modal.App("kidney-numpy-training")

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch", "torchvision", "torchaudio", 
        "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn",
        "tqdm", "tensorboard", "opencv-python-headless"
    ])
)

# Create volume for training data
volume = modal.Volume.from_name("kidney-training-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/training_data": volume},
    timeout=3600  # 1 hour timeout
)
def upload_training_data(files_data):
    """Upload local numpy training data to Modal volume"""
    import os
    
    print("📂 Uploading training data to Modal...")
    
    # Copy files to volume
    volume_path = "/training_data"
    os.makedirs(volume_path, exist_ok=True)
    
    uploaded_count = 0
    for filename, data in files_data.items():
        remote_file = os.path.join(volume_path, filename)
        
        # Write file content
        with open(remote_file, 'wb') as f:
            f.write(data)
        print(f"   ✅ Uploaded {filename} ({len(data)} bytes)")
        uploaded_count += 1
    
    # Commit the volume to persist data
    volume.commit()
    print("📂 Upload complete and committed!")
    return uploaded_count

# UNet3D Model Definition
@app.function(image=image)
def get_unet_model():
    """Return UNet3D model definition"""
    import torch
    import torch.nn as nn
    
    class UNet3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
            super(UNet3D, self).__init__()
            
            # Encoder
            self.encoder = nn.ModuleList()
            self.encoder_pools = nn.ModuleList()
            
            for feature in features:
                self.encoder.append(self._make_conv_block(in_channels, feature))
                self.encoder_pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
                in_channels = feature
            
            # Bottleneck
            self.bottleneck = self._make_conv_block(features[-1], features[-1] * 2)
            
            # Decoder
            self.decoder = nn.ModuleList()
            self.decoder_upconvs = nn.ModuleList()
            
            for feature in reversed(features):
                self.decoder_upconvs.append(
                    nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
                )
                self.decoder.append(self._make_conv_block(feature * 2, feature))
            
            # Final layer
            self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
        
        def _make_conv_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            # Encoder
            skip_connections = []
            for encoder, pool in zip(self.encoder, self.encoder_pools):
                x = encoder(x)
                skip_connections.append(x)
                x = pool(x)
            
            # Bottleneck
            x = self.bottleneck(x)
            
            # Decoder
            skip_connections = skip_connections[::-1]
            for idx, (upconv, decoder) in enumerate(zip(self.decoder_upconvs, self.decoder)):
                x = upconv(x)
                skip_connection = skip_connections[idx]
                
                # Handle size mismatch
                if x.shape != skip_connection.shape:
                    x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
                
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = decoder(concat_skip)
            
            # Final output
            x = self.final_conv(x)
            x = self.sigmoid(x)
            return x
    
    return UNet3D

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/training_data": volume},
    timeout=7200  # 2 hour timeout
)
def train_kidney_model():
    """Train kidney detection model on Modal A10 GPU"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import pickle
    from tqdm import tqdm
    import os
    
    print("🚀 Starting Modal A10 GPU training...")
    print(f"   🔧 Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Reload volume to ensure we have the latest data
    volume.reload()
    
    # List files in training data directory to debug
    training_data_path = "/training_data"
    if os.path.exists(training_data_path):
        files = os.listdir(training_data_path)
        print(f"   📁 Files in training_data: {files}")
    else:
        print(f"   ❌ Training data directory not found: {training_data_path}")
        return {"error": "Training data not found"}
    
    # Dataset class
    class KidneyDataset(Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            
            # Load metadata if exists, otherwise skip
            metadata_path = os.path.join(data_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                self.metadata = {}
                print("   ⚠️  No metadata found, continuing without it")
            
            # Find all MRI files
            self.samples = []
            for i in range(7):  # We have 7 samples
                mri_file = f"mri_{i:03d}.npy"
                mask_file = f"mask_{i:03d}.npy"
                
                mri_path = os.path.join(data_dir, mri_file)
                mask_path = os.path.join(data_dir, mask_file)
                
                if os.path.exists(mri_path) and os.path.exists(mask_path):
                    self.samples.append((mri_path, mask_path))
            
            print(f"   📊 Loaded {len(self.samples)} training samples")
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            mri_path, mask_path = self.samples[idx]
            
            # Load data
            mri = np.load(mri_path)
            mask = np.load(mask_path)
            
            # Convert to tensors
            mri_tensor = torch.FloatTensor(mri).unsqueeze(0)  # Add channel dimension
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
            
            return mri_tensor, mask_tensor
    
    # Combined loss function
    class CombinedLoss(nn.Module):
        def __init__(self, bce_weight=0.5, dice_weight=0.5):
            super(CombinedLoss, self).__init__()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.bce_loss = nn.BCELoss()
        
        def dice_loss(self, pred, target, smooth=1e-6):
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return 1 - dice
        
        def forward(self, pred, target):
            bce = self.bce_loss(pred, target)
            dice = self.dice_loss(pred, target)
            return self.bce_weight * bce + self.dice_weight * dice
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🔧 Training device: {device}")
    
    # Load dataset
    dataset = KidneyDataset("/training_data")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Small batch for GPU memory
    
    # Initialize model
    UNet3D = get_unet_model.remote()
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Training loop
    num_epochs = 300
    best_loss = float('inf')
    
    print(f"   🏃 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (mri, mask) in enumerate(progress_bar):
            mri, mask = mri.to(device), mask.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(mri)
            loss = criterion(outputs, mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg': f'{epoch_loss/(batch_idx+1):.6f}'
            })
        
        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"   📈 Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save model after each epoch with loss in filename
        epoch_model_path = f'/training_data/kidney_model_epoch_{epoch+1:03d}_loss_{avg_loss:.6f}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_loss,
            'model_config': {
                'in_channels': 1,
                'out_channels': 1,
                'features': [64, 128, 256, 512]
            }
        }, epoch_model_path)
        
        print(f"   💾 Epoch model saved: epoch_{epoch+1:03d}_loss_{avg_loss:.6f}.pth")
        
        # Save best model (keep this for backwards compatibility)
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Save best model
            best_model_path = '/training_data/kidney_model_modal_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_loss,
                'model_config': {
                    'in_channels': 1,
                    'out_channels': 1,
                    'features': [64, 128, 256, 512]
                }
            }, best_model_path)
            
            print(f"   🏆 New best model saved! Loss: {best_loss:.6f}")
        
        # Early stopping check
        if epoch > 50 and avg_loss < 0.01:
            print(f"   ✅ Early stopping triggered! Loss below threshold.")
            break
    
    print(f"   🎉 Training complete! Best loss: {best_loss:.6f}")
    
    # Final model info
    final_model_path = '/training_data/kidney_model_modal_best.pth'
    volume.commit()  # Commit all trained models
    
    model_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)
    print(f"   📦 Best model saved: {final_model_path} ({model_size_mb:.1f} MB)")
    
    # List all epoch models created
    epoch_models = [f for f in os.listdir('/training_data') if f.startswith('kidney_model_epoch_')]
    epoch_models.sort()
    print(f"   📁 Total epoch models saved: {len(epoch_models)}")
    for model_file in epoch_models[:5]:  # Show first 5
        print(f"      {model_file}")
    if len(epoch_models) > 5:
        print(f"      ... and {len(epoch_models) - 5} more")
    
    return {
        'best_loss': best_loss,
        'epochs_trained': epoch + 1,
        'model_path': final_model_path,
        'model_size_mb': model_size_mb,
        'total_epoch_models': len(epoch_models)
    }

@app.function(
    image=image,
    volumes={"/training_data": volume},
    timeout=600
)
def download_trained_model():
    """Download the trained model from Modal volume"""
    import os
    
    model_path = '/training_data/kidney_model_modal_best.pth'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        print(f"   📦 Model ready for download ({len(model_data)} bytes)")
        return model_data
    else:
        raise FileNotFoundError("Trained model not found!")

@app.function(
    image=image,
    volumes={"/training_data": volume},
    timeout=300
)
def list_epoch_models():
    """List all epoch models with their losses"""
    import os
    
    epoch_models = []
    training_dir = '/training_data'
    
    if os.path.exists(training_dir):
        files = os.listdir(training_dir)
        for filename in files:
            if filename.startswith('kidney_model_epoch_') and filename.endswith('.pth'):
                # Extract epoch and loss from filename
                # Format: kidney_model_epoch_XXX_loss_Y.YYYYYY.pth
                try:
                    parts = filename.replace('.pth', '').split('_')
                    epoch_num = int(parts[3])  # epoch number
                    loss_value = float(parts[5])  # loss value
                    
                    file_path = os.path.join(training_dir, filename)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    
                    epoch_models.append({
                        'filename': filename,
                        'epoch': epoch_num,
                        'loss': loss_value,
                        'size_mb': file_size
                    })
                except (IndexError, ValueError):
                    continue
    
    # Sort by epoch number
    epoch_models.sort(key=lambda x: x['epoch'])
    
    return epoch_models

@app.function(
    image=image,
    volumes={"/training_data": volume},
    timeout=600
)
def download_epoch_model(filename):
    """Download a specific epoch model by filename"""
    import os
    
    model_path = f'/training_data/{filename}'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = f.read()
        print(f"   📦 Epoch model ready for download: {filename} ({len(model_data)} bytes)")
        return model_data
    else:
        raise FileNotFoundError(f"Epoch model not found: {filename}")

@app.local_entrypoint()
def main():
    """Main training pipeline"""
    print("🚀 Modal Kidney Training Pipeline")
    print("=" * 50)
    
    # Step 1: Read local training data
    print("\n📂 Step 1: Reading local training data...")
    local_data_dir = r"c:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\training\kidneyTrainingData"
    
    # Files to upload
    files_to_upload = [
        "mri_000.npy", "mri_001.npy", "mri_002.npy", "mri_003.npy", 
        "mri_004.npy", "mri_005.npy", "mri_006.npy",
        "mask_000.npy", "mask_001.npy", "mask_002.npy", "mask_003.npy",
        "mask_004.npy", "mask_005.npy", "mask_006.npy",
        "metadata.pkl"
    ]
    
    # Read all files into memory
    files_data = {}
    for filename in files_to_upload:
        local_file = os.path.join(local_data_dir, filename)
        
        if os.path.exists(local_file):
            with open(local_file, 'rb') as f:
                files_data[filename] = f.read()
            print(f"   📁 Read {filename} ({len(files_data[filename])} bytes)")
        else:
            print(f"   ❌ Missing {filename}")
    
    print(f"   📊 Total files to upload: {len(files_data)}")
    
    # Step 2: Upload training data
    print("\n📂 Step 2: Uploading training data...")
    upload_result = upload_training_data.remote(files_data)
    print(f"   ✅ Uploaded {upload_result} files")
    
    # Step 3: Train model
    print("\n🏃 Step 3: Training model on A10 GPU...")
    training_result = train_kidney_model.remote()
    print(f"   🎉 Training complete!")
    print(f"   📊 Best loss: {training_result['best_loss']:.6f}")
    print(f"   ⏱️  Epochs: {training_result['epochs_trained']}")
    print(f"   📦 Model size: {training_result['model_size_mb']:.1f} MB")
    print(f"   📁 Total epoch models: {training_result['total_epoch_models']}")
    
    # Step 4: List epoch models
    print("\n📋 Step 4: Listing epoch models...")
    epoch_models = list_epoch_models.remote()
    
    if epoch_models:
        print(f"   📊 Found {len(epoch_models)} epoch models:")
        print("   Epoch | Loss     | Size (MB) | Filename")
        print("   ------|----------|-----------|----------")
        for model in epoch_models:
            print(f"   {model['epoch']:5d} | {model['loss']:8.6f} | {model['size_mb']:9.1f} | {model['filename']}")
        
        # Show trend analysis
        if len(epoch_models) >= 3:
            early_loss = epoch_models[2]['loss']  # Loss at epoch 3
            best_loss = min(m['loss'] for m in epoch_models)
            best_epoch = next(m['epoch'] for m in epoch_models if m['loss'] == best_loss)
            
            print(f"\n   📈 Training Analysis:")
            print(f"      Early loss (epoch 3): {early_loss:.6f}")
            print(f"      Best loss: {best_loss:.6f} (epoch {best_epoch})")
            print(f"      Improvement: {((early_loss - best_loss) / early_loss * 100):.1f}%")
            
            # Check for overfitting (loss increasing after best)
            models_after_best = [m for m in epoch_models if m['epoch'] > best_epoch]
            if models_after_best:
                worst_after_best = max(m['loss'] for m in models_after_best)
                if worst_after_best > best_loss * 1.1:  # 10% increase indicates overfitting
                    print(f"      ⚠️  Potential overfitting detected after epoch {best_epoch}")
                    print(f"      💡 Consider using model from epoch {best_epoch}")
    
    # Step 5: Download best trained model
    print("\n📥 Step 5: Downloading best trained model...")
    model_data = download_trained_model.remote()
    
    # Save locally
    local_model_path = r"c:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\training\kidney_model_modal_numpy.pth"
    with open(local_model_path, 'wb') as f:
        f.write(model_data)
    
    print(f"   💾 Best model saved locally: {local_model_path}")
    
    # Optional: Download specific epoch model
    if epoch_models:
        print(f"\n📝 To download a specific epoch model, use:")
        print(f"   modal run modal_numpy_training.py::download_epoch_model --filename <model_filename>")
        print(f"   Example: modal run modal_numpy_training.py::download_epoch_model --filename {epoch_models[0]['filename']}")
    
    print("\n✅ Pipeline complete!")

if __name__ == "__main__":
    main()
