"""
Fresh AI Kidney Detection Training - New Model
==============================================
Train a completely new kidney detection model from scratch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from datetime import datetime
from unet_3d import UNet3D

class FreshKidneyDataset(Dataset):
    """Fresh dataset loader for kidney training"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.load_data()
    
    def load_data(self):
        """Load all training data"""
        print("🔍 Loading fresh training data...")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.mat'):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
                    if 'images' in data:
                        images = data['images']
                        print(f"\n🗂️  {filename} contains {len(images)} images:")
                        for i in range(len(images)):
                            img = images[i]
                            name = ''
                            if hasattr(img, 'Name'):
                                if isinstance(img.Name, str):
                                    name = img.Name
                                else:
                                    try:
                                        name = ''.join(chr(c) for c in img.Name.flatten() if c != 0)
                                    except:
                                        name = f"img_{i}"
                            shape = getattr(img, 'data', np.array([])).shape if hasattr(img, 'data') else None
                            print(f"   - {name} {shape}")
                        # Find MRI and kidney mask slaves
                        for i in range(len(images)):
                            img = images[i]
                            name = ''
                            if hasattr(img, 'Name'):
                                if isinstance(img.Name, str):
                                    name = img.Name
                                else:
                                    try:
                                        name = ''.join(chr(c) for c in img.Name.flatten() if c != 0)
                                    except:
                                        name = f"img_{i}"
                            if 'mri' in name.lower() and hasattr(img, 'data') and len(img.data.shape) == 3:
                                mri_data = img.data
                                print(f"   📷 Found MRI: {name} {img.data.shape}")
                                # Look for slaves field
                                if hasattr(img, 'slaves') and isinstance(img.slaves, np.ndarray):
                                    for slave in img.slaves:
                                        slave_name = ''
                                        if hasattr(slave, 'Name'):
                                            if isinstance(slave.Name, str):
                                                slave_name = slave.Name
                                            else:
                                                try:
                                                    slave_name = ''.join(chr(c) for c in slave.Name.flatten() if c != 0)
                                                except:
                                                    slave_name = 'slave'
                                        # Only use slaves with 'Kidney' in name and not 'SRF'
                                        if 'kidney' not in slave_name.lower() or 'srf' in slave_name.lower():
                                            print(f"      ⚠️  Skipped slave: {slave_name} (not a valid kidney mask)")
                                            continue
                                        if not hasattr(slave, 'data') or not isinstance(slave.data, np.ndarray) or len(slave.data.shape) != 3:
                                            print(f"      ⚠️  Skipped slave: {slave_name} (not 3D)")
                                            continue
                                        mask = slave.data
                                        if mask.shape != mri_data.shape:
                                            print(f"      ⚠️  Skipped slave: {slave_name} (shape mismatch {mask.shape} vs MRI {mri_data.shape})")
                                            continue
                                        self.samples.append({
                                            'mri': mri_data.astype(np.float32),
                                            'kidney': (mask > 0).astype(np.float32),
                                            'file': filename,
                                            'slave_name': slave_name
                                        })
                                        print(f"      ✅ Added sample: {filename} (slave: {slave_name})")
                except Exception as e:
                    print(f"   ⚠️  Skipped {filename}: {str(e)}")
        
        print(f"📊 Total training samples: {len(self.samples)}")
        for i, sample in enumerate(self.samples):
            print(f"   {i+1}. {sample['file']} - slave: {sample['slave_name']}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get MRI and mask
        mri = sample['mri'].copy()
        mask = sample['kidney'].copy()
        
        # Normalize MRI using percentiles
        p1, p99 = np.percentile(mri, [1, 99])
        mri = (mri - p1) / (p99 - p1)
        mri = np.clip(mri, 0, 1)
        
        # Resize to training size
        target_shape = (64, 64, 32)
        zoom_factors = [target_shape[i] / mri.shape[i] for i in range(3)]
        
        mri_resized = zoom(mri, zoom_factors, order=1)
        mask_resized = zoom(mask, zoom_factors, order=0)  # Nearest neighbor for mask
        
        # Convert to tensors
        mri_tensor = torch.from_numpy(mri_resized).float().unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
        
        return mri_tensor, mask_tensor

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined loss: BCE + Dice + Focal"""
    
    def __init__(self, bce_weight=0.3, dice_weight=0.4, focal_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        total_loss = (self.bce_weight * bce + 
                     self.dice_weight * dice + 
                     self.focal_weight * focal)
        
        return total_loss, bce, dice, focal

def train_fresh_model():
    """Train a completely fresh kidney detection model"""
    print("🚀 TRAINING FRESH AI KIDNEY DETECTION MODEL")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Load dataset
    data_dir = '../../../data/training'
    if not os.path.exists(data_dir):
        data_dir = '../../data/training'
        if not os.path.exists(data_dir):
            data_dir = 'training'
    
    dataset = FreshKidneyDataset(data_dir)
    
    if len(dataset) == 0:
        raise ValueError("No training data found!")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize fresh model
    model = UNet3D(in_channels=1, out_channels=1)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Fresh model parameters: {total_params:,}")
    
    # Setup training
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    num_epochs = 100
    best_loss = float('inf')
    
    print(f"\n🏋️ Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_dice = 0.0
        epoch_focal = 0.0
        
        for batch_idx, (mri, mask) in enumerate(dataloader):
            mri, mask = mri.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(mri)
            total_loss, bce_loss, dice_loss, focal_loss = criterion(outputs, mask)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_bce += bce_loss.item()
            epoch_dice += dice_loss.item()
            epoch_focal += focal_loss.item()
        
        # Average losses
        avg_loss = epoch_loss / len(dataloader)
        avg_bce = epoch_bce / len(dataloader)
        avg_dice = epoch_dice / len(dataloader)
        avg_focal = epoch_focal / len(dataloader)
        
        # Update learning rate
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"BCE: {avg_bce:.4f} | "
                  f"Dice: {avg_dice:.4f} | "
                  f"Focal: {avg_focal:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'kidney_fresh_model_{timestamp}.pth'
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'training_samples': len(dataset)
            }, model_path)
            
            # Also save as the "best" model for inference
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'training_samples': len(dataset)
            }, 'kidney_unet_model_best.pth')
            
            print(f"   💾 New best model saved! Loss: {best_loss:.4f}")
    
    print(f"\n✅ Training completed!")
    print(f"📊 Final best loss: {best_loss:.4f}")
    print(f"🎯 Model saved as: kidney_unet_model_best.pth")
    print(f"📈 Training samples used: {len(dataset)}")

if __name__ == "__main__":
    train_fresh_model()
