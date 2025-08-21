"""
Direct Modal Kidney Training with Improved Logic
===============================================
"""

import modal

app = modal.App("kidney-training-improved")

# Volumes
data_vol = modal.Volume.from_name("kidneyDrawing", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("kidneyCheckpoints", create_if_missing=True)

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
    }
)
def train_improved_on_modal():
    """Run improved kidney training directly on Modal A10 GPU"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import scipy.io as sio
    import os
    from torch.utils.data import Dataset, DataLoader
    from scipy.ndimage import zoom
    from datetime import datetime
    
    print("🚀 Modal AI Improved Kidney Training Started!")
    print("=" * 60)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Check training data
    data_files = [f for f in os.listdir("/data/training") if f.endswith('.mat')]
    print(f"📁 Found {len(data_files)} training files: {data_files}")
    
    if not data_files:
        return "❌ No training data found!"
    
    # UNet3D model definition
    class UNet3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, init_features=32):
            super(UNet3D, self).__init__()
            
            features = init_features
            self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
            self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
            self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")
            
            self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
            self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
            self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
            self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
            self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
            self.decoder1 = UNet3D._block(features * 2, features, name="dec1")
            
            self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        def forward(self, x):
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))
            
            bottleneck = self.bottleneck(self.pool4(enc4))
            
            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            
            return self.conv(dec1)
        
        @staticmethod
        def _block(in_channels, features, name):
            return nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(num_features=features),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(num_features=features),
                nn.ReLU(inplace=True)
            )
    
    # Improved Dataset with our new logic
    class ImprovedKidneyDataset(Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.samples = []
            self.load_data()
        
        def load_data(self):
            print("🔍 Loading training data with improved extraction...")
            
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.mat'):
                    file_path = os.path.join(self.data_dir, filename)
                    try:
                        data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
                        if 'images' in data:
                            images = data['images']
                            print(f"📁 {filename}: {len(images)} images")
                            self.extract_kidney_samples(images, filename)
                    except Exception as e:
                        print(f"⚠️ Skipped {filename}: {str(e)}")
            
            print(f"📊 Total training samples: {len(self.samples)}")
            for i, sample in enumerate(self.samples):
                coverage = np.sum(sample['kidney']) / np.prod(sample['kidney'].shape) * 100
                print(f"   {i+1}. {sample['file']} - {sample['slave_name']} ({coverage:.1f}%)")
        
        def extract_kidney_samples(self, images, filename):
            for i in range(len(images)):
                img = images[i]
                name = self.get_image_name(img, i)
                
                if ('mri' in name.lower() and hasattr(img, 'data') and 
                    len(img.data.shape) == 3):
                    mri_data = img.data
                    print(f"   🧠 Found MRI: {name} {mri_data.shape}")
                    
                    # Look for kidney slaves with improved logic
                    if hasattr(img, 'slaves') and isinstance(img.slaves, np.ndarray):
                        for slave in img.slaves:
                            slave_name = self.get_image_name(slave, 0)
                            slave_name_lower = slave_name.lower()
                            
                            # Include ANY slave with "kidney" but exclude "SRF"
                            if ('kidney' in slave_name_lower and 
                                'srf' not in slave_name_lower and
                                hasattr(slave, 'data') and 
                                isinstance(slave.data, np.ndarray) and 
                                len(slave.data.shape) == 3 and
                                slave.data.shape == mri_data.shape):
                                
                                mask = slave.data
                                kidney_coverage = np.sum(mask > 0) / np.prod(mask.shape) * 100
                                
                                # Add ALL kidney slaves regardless of coverage
                                self.samples.append({
                                    'mri': mri_data.astype(np.float32),
                                    'kidney': (mask > 0).astype(np.float32),
                                    'file': filename,
                                    'slave_name': slave_name
                                })
                                print(f"      ✅ Added: {slave_name} ({kidney_coverage:.1f}%)")
        
        def get_image_name(self, img, index):
            if hasattr(img, 'Name') and img.Name is not None:
                if isinstance(img.Name, str):
                    return img.Name
                else:
                    try:
                        return ''.join(chr(c) for c in img.Name.flatten() if c != 0 and 32 <= c <= 126)
                    except:
                        return f"img_{index}"
            return f"img_{index}"
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            mri = sample['mri'].copy()
            mask = sample['kidney'].copy()
            
            # Improved normalization
            p1, p99 = np.percentile(mri[mri > 0], [1, 99])
            if p99 > p1:
                mri = (mri - p1) / (p99 - p1)
                mri = np.clip(mri, 0, 1)
            else:
                mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
            
            # Resize to training size
            target_shape = (64, 64, 32)
            zoom_factors = [target_shape[i] / mri.shape[i] for i in range(3)]
            
            mri_resized = zoom(mri, zoom_factors, order=1)
            mask_resized = zoom(mask, zoom_factors, order=0)
            
            # Convert to tensors
            mri_tensor = torch.from_numpy(mri_resized).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
            
            return mri_tensor, mask_tensor
    
    # Loss functions
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1.0):
            super(DiceLoss, self).__init__()
            self.smooth = smooth
            
        def forward(self, inputs, targets):
            inputs = torch.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            intersection = (inputs * targets).sum()
            dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
            return 1 - dice
    
    class CombinedLoss(nn.Module):
        def __init__(self):
            super(CombinedLoss, self).__init__()
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
            self.dice_loss = DiceLoss()
        
        def forward(self, inputs, targets):
            bce = self.bce_loss(inputs, targets)
            dice = self.dice_loss(inputs, targets)
            total_loss = 0.4 * bce + 0.6 * dice
            return total_loss, bce, dice
    
    # Load dataset
    dataset = ImprovedKidneyDataset("/data/training")
    if len(dataset) == 0:
        return "❌ No training samples found!"
    
    # Create DataLoader
    batch_size = min(4, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,}")
    
    # Setup training
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)
    
    # Training loop
    num_epochs = 200
    best_loss = float('inf')
    
    print(f"🏋️ Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_dice = 0.0
        
        for batch_idx, (mri, mask) in enumerate(dataloader):
            mri, mask = mri.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(mri)
            total_loss, bce_loss, dice_loss = criterion(outputs, mask)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_bce += bce_loss.item()
            epoch_dice += dice_loss.item()
        
        # Average losses
        avg_loss = epoch_loss / len(dataloader)
        avg_bce = epoch_bce / len(dataloader)
        avg_dice = epoch_dice / len(dataloader)
        
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"BCE: {avg_bce:.4f} | "
                  f"Dice: {avg_dice:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        # Save checkpoint every 20 epochs and best model
        if epoch % 20 == 0 or avg_loss < best_loss:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'training_samples': len(dataset),
                'model_info': 'Improved kidney detection with fixed data extraction'
            }
            
            # Save regular checkpoint
            if epoch % 20 == 0:
                torch.save(checkpoint, f'/checkpoints/kidney_improved_checkpoint_epoch_{epoch+1}.pth')
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, '/checkpoints/kidney_unet_model_improved_final.pth')
                print(f"   💾 New best model saved! Loss: {best_loss:.4f}")
    
    final_result = {
        'status': 'completed',
        'final_loss': best_loss,
        'epochs_completed': num_epochs,
        'training_samples': len(dataset),
        'model_path': '/checkpoints/kidney_unet_model_improved_final.pth'
    }
    
    print(f"✅ Training completed!")
    print(f"📊 Final best loss: {best_loss:.4f}")
    print(f"📈 Training samples used: {len(dataset)}")
    
    return final_result

@app.local_entrypoint()
def main():
    """Run the improved kidney training"""
    print("🚀 Starting improved kidney training on Modal A10 GPU...")
    
    with app.run():
        result = train_improved_on_modal.remote()
        
        print(f"✅ Training completed!")
        print(f"📊 Results: {result}")
        print("💾 Download model: modal volume get kidneyCheckpoints kidney_unet_model_improved_final.pth")

if __name__ == "__main__":
    main()
