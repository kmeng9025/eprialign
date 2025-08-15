#!/usr/bin/env python3
"""
Improved Kidney Detection Training
=================================

This script addresses the high loss issue by using:
- Focal Loss for class imbalance
- Combined BCE + Dice Loss
- Better learning rate scheduling
- Improved data preprocessing

Author: AI Assistant
Date: 2025-08-14
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings('ignore')

# Import model architecture
from unet_3d import UNet3D

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Compute focal loss
        ce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and dice
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE + Dice + Focal Loss"""
    
    def __init__(self, bce_weight=0.3, dice_weight=0.4, focal_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2)
        
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        total_loss = (self.bce_weight * bce + 
                     self.dice_weight * dice + 
                     self.focal_weight * focal)
        
        return total_loss, bce, dice, focal

class ImprovedKidneyDataset(Dataset):
    """Improved dataset with better preprocessing"""
    
    def __init__(self, data_dir, augment=True, normalize_kidneys=True):
        self.data_dir = data_dir
        self.augment = augment
        self.normalize_kidneys = normalize_kidneys
        self.samples = []
        
        print(f"🔍 Loading training data from: {data_dir}")
        
        # Find all .mat files in training directory
        mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
        print(f"   📁 Found {len(mat_files)} .mat files")
        
        for mat_file in mat_files:
            self._process_mat_file(mat_file)
        
        print(f"   ✅ Loaded {len(self.samples)} training samples")
        
        if len(self.samples) == 0:
            raise ValueError("No valid training samples found!")
    
    def _process_mat_file(self, mat_file):
        """Process a single .mat file to extract training samples"""
        print(f"   📂 Processing: {os.path.basename(mat_file)}")
        
        # Load the .mat file
        data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        
        if 'images' not in data:
            print(f"      ❌ No 'images' field found")
            return
        
        images = data['images']
        if not hasattr(images, '__len__'):
            images = [images]
        
        # Find MRI images and their corresponding kidney slaves
        for i, img in enumerate(images):
            if not hasattr(img, 'data') or img.data is None:
                continue
            
            # Check if this is an MRI image
            image_name = ""
            if hasattr(img, 'Name') and img.Name is not None:
                if isinstance(img.Name, str):
                    image_name = img.Name
                elif hasattr(img.Name, '__len__'):
                    try:
                        image_name = ''.join(chr(c) for c in img.Name.flatten() if c != 0)
                    except:
                        image_name = str(img.Name)
            
            # Check if it's an MRI image
            shape = img.data.shape if hasattr(img.data, 'shape') else None
            is_mri = False
            
            if shape and len(shape) == 3:
                if "mri" in image_name.lower():
                    is_mri = True
                elif shape[0] == 350 and shape[1] == 350:  # Common MRI dimensions
                    is_mri = True
            
            if not is_mri:
                continue
            
            print(f"      🧠 Found MRI: {image_name} {shape}")
            
            # Look for kidney slaves in this image
            kidney_mask = None
            if hasattr(img, 'slaves') and img.slaves is not None:
                if not hasattr(img.slaves, '__len__'):
                    slaves = [img.slaves]
                else:
                    slaves = img.slaves
                
                # Combine all kidney slaves into one mask
                for slave in slaves:
                    if hasattr(slave, 'Name') and slave.Name is not None:
                        slave_name = ""
                        if isinstance(slave.Name, str):
                            slave_name = slave.Name
                        elif hasattr(slave.Name, '__len__'):
                            try:
                                slave_name = ''.join(chr(c) for c in slave.Name.flatten() if c != 0)
                            except:
                                slave_name = str(slave.Name)
                        
                        # Check if this is a kidney slave - BUT ONLY VOLUME DATA, NOT SURFACE
                        if "kidney" in slave_name.lower() and not "srf" in slave_name.lower():
                            print(f"         🫘 Found kidney volume slave: {slave_name}")
                            
                            if hasattr(slave, 'data') and slave.data is not None:
                                try:
                                    # Handle different data types
                                    slave_data = slave.data
                                    
                                    # Convert to numpy array if needed
                                    if hasattr(slave_data, 'astype'):
                                        slave_array = slave_data.astype(bool)
                                    else:
                                        slave_array = np.array(slave_data, dtype=bool)
                                    
                                    # Verify this is actually a proper 3D mask
                                    if len(slave_array.shape) == 3 and slave_array.shape == img.data.shape:
                                        # Check that it's not all ones or all zeros
                                        unique_vals = np.unique(slave_array)
                                        if len(unique_vals) == 2 and True in unique_vals and False in unique_vals:
                                            kidney_voxels = np.sum(slave_array)
                                            total_voxels = np.prod(slave_array.shape)
                                            coverage_percent = kidney_voxels / total_voxels * 100
                                            
                                            # Only accept masks with reasonable coverage (0.1% to 50%)
                                            if 0.1 <= coverage_percent <= 50.0:
                                                if kidney_mask is None:
                                                    kidney_mask = slave_array
                                                else:
                                                    kidney_mask |= slave_array
                                                print(f"         ✅ Valid kidney mask: {kidney_voxels:,} voxels ({coverage_percent:.2f}%)")
                                            else:
                                                print(f"         ⚠️  Rejected mask - unusual coverage: {coverage_percent:.2f}%")
                                        else:
                                            print(f"         ⚠️  Rejected mask - not binary: {unique_vals}")
                                    else:
                                        print(f"         ⚠️  Rejected mask - wrong shape: {slave_array.shape} vs {img.data.shape}")
                                        
                                except Exception as e:
                                    print(f"         ⚠️  Could not process slave data: {e}")
                                    continue
                        elif "kidney" in slave_name.lower() and "srf" in slave_name.lower():
                            print(f"         🚫 Skipping kidney surface slave: {slave_name} (not volume data)")
            
            if kidney_mask is not None:
                kidney_count = np.sum(kidney_mask)
                kidney_percent = kidney_count / np.prod(kidney_mask.shape) * 100
                print(f"         ✅ Final kidney mask: {kidney_count} voxels ({kidney_percent:.2f}%)")
                
                # Ensure MRI data is also properly converted
                try:
                    mri_array = np.array(img.data, dtype=np.float32)
                    
                    # Add this sample
                    self.samples.append({
                        'mri_data': mri_array,
                        'kidney_mask': kidney_mask.astype(np.float32),
                        'source_file': os.path.basename(mat_file),
                        'image_name': image_name,
                        'kidney_percent': kidney_percent
                    })
                except Exception as e:
                    print(f"         ⚠️  Could not process MRI data: {e}")
            else:
                print(f"         ❌ No kidney slaves found")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        mri_data = sample['mri_data'].copy()
        kidney_mask = sample['kidney_mask'].copy()
        
        # Improved normalization
        # Normalize MRI data with robust statistics (exclude outliers)
        mri_flat = mri_data.flatten()
        p1, p99 = np.percentile(mri_flat, [1, 99])
        mri_clipped = np.clip(mri_data, p1, p99)
        mri_data = (mri_clipped - p1) / (p99 - p1 + 1e-8)
        
        # Resize to target size for U-Net
        target_size = (64, 64, 32)
        
        from scipy.ndimage import zoom
        
        zoom_factors = [t/s for t, s in zip(target_size, mri_data.shape)]
        mri_resized = zoom(mri_data, zoom_factors, order=1)
        mask_resized = zoom(kidney_mask, zoom_factors, order=0)  # Nearest neighbor for masks
        
        # Data augmentation (if enabled)
        if self.augment:
            # Random rotation/flipping
            if np.random.random() > 0.5:
                # Flip along x-axis
                mri_resized = mri_resized[::-1, :, :].copy()
                mask_resized = mask_resized[::-1, :, :].copy()
            
            if np.random.random() > 0.5:
                # Flip along y-axis
                mri_resized = mri_resized[:, ::-1, :].copy()
                mask_resized = mask_resized[:, ::-1, :].copy()
            
            # Random intensity scaling
            intensity_factor = np.random.uniform(0.7, 1.3)
            mri_resized = np.clip(mri_resized * intensity_factor, 0, 1)
            
            # Random noise
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 0.01, mri_resized.shape)
                mri_resized = np.clip(mri_resized + noise, 0, 1)
        
        # Ensure arrays are contiguous
        mri_resized = np.ascontiguousarray(mri_resized)
        mask_resized = np.ascontiguousarray(mask_resized)
        
        # Convert to tensors
        mri_tensor = torch.FloatTensor(mri_resized).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.FloatTensor(mask_resized).unsqueeze(0)  # Add channel dimension for consistency
        
        return mri_tensor, mask_tensor

def train_improved_model():
    """Main training function with improved loss and optimization"""
    
    print("🚀 IMPROVED KIDNEY DETECTION MODEL TRAINING")
    print("="*60)
    print("📋 Using: Focal Loss + Dice Loss + Improved Preprocessing")
    
    # Configuration
    config = {
        'data_dir': r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training",
        'model_save_path': r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\kidney_unet_model_best.pth",
        'batch_size': 2,
        'num_epochs': 50,  # More epochs for better convergence
        'learning_rate': 5e-4,  # Higher initial learning rate
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"📂 Training data directory: {config['data_dir']}")
    print(f"💾 Model save path: {config['model_save_path']}")
    print(f"🔧 Device: {config['device']}")
    print(f"📊 Batch size: {config['batch_size']}")
    print(f"🔄 Epochs: {config['num_epochs']}")
    print(f"📈 Learning rate: {config['learning_rate']}")
    
    # Create dataset and dataloader
    print("\n📚 Creating improved training dataset...")
    dataset = ImprovedKidneyDataset(config['data_dir'], augment=True)
    
    # Print dataset statistics
    kidney_percentages = [sample['kidney_percent'] for sample in dataset.samples]
    print(f"   📊 Kidney coverage range: {min(kidney_percentages):.2f}% - {max(kidney_percentages):.2f}%")
    print(f"   📊 Average kidney coverage: {np.mean(kidney_percentages):.2f}%")
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"   ✅ Training batches: {len(dataloader)}")
    
    # Initialize model
    print("\n🧠 Initializing U-Net model...")
    model = UNet3D(in_channels=1, out_channels=1)
    model.to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   🔧 Trainable parameters: {trainable_params:,}")
    
    # Initialize improved loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.4, focal_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n🚀 Starting improved training for {config['num_epochs']} epochs...")
    print("="*60)
    
    best_loss = float('inf')
    history = {'loss': [], 'bce': [], 'dice': [], 'focal': [], 'dice_score': [], 'iou': []}
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_dice = 0.0
        epoch_focal = 0.0
        epoch_dice_score = 0.0
        epoch_iou = 0.0
        
        print(f"\n📅 Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(config['device']), target.to(config['device'])
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            total_loss, bce_loss, dice_loss, focal_loss = criterion(output, target)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(output)
                pred_binary = (pred_sigmoid > 0.5).float()
                
                # Dice score
                intersection = (pred_binary * target).sum()
                dice_score = (2.0 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)
                
                # IoU score
                union = pred_binary.sum() + target.sum() - intersection
                iou_score = intersection / (union + 1e-8)
            
            # Accumulate metrics
            epoch_loss += total_loss.item()
            epoch_bce += bce_loss.item()
            epoch_dice += dice_loss.item()
            epoch_focal += focal_loss.item()
            epoch_dice_score += dice_score.item()
            epoch_iou += iou_score.item()
            
            print(f"   📊 Batch {batch_idx+1}/{len(dataloader)}: "
                  f"Loss={total_loss.item():.4f}, "
                  f"BCE={bce_loss.item():.4f}, "
                  f"Dice={dice_loss.item():.4f}, "
                  f"Focal={focal_loss.item():.4f}, "
                  f"DiceScore={dice_score.item():.4f}, "
                  f"IoU={iou_score.item():.4f}")
        
        # Calculate epoch averages
        avg_loss = epoch_loss / len(dataloader)
        avg_bce = epoch_bce / len(dataloader)
        avg_dice = epoch_dice / len(dataloader)
        avg_focal = epoch_focal / len(dataloader)
        avg_dice_score = epoch_dice_score / len(dataloader)
        avg_iou = epoch_iou / len(dataloader)
        
        # Update learning rate
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   🎯 Epoch {epoch+1} Results:")
        print(f"      Total Loss: {avg_loss:.4f}")
        print(f"      BCE Loss: {avg_bce:.4f}")
        print(f"      Dice Loss: {avg_dice:.4f}")
        print(f"      Focal Loss: {avg_focal:.4f}")
        print(f"      Dice Score: {avg_dice_score:.4f}")
        print(f"      IoU Score: {avg_iou:.4f}")
        print(f"      LR: {current_lr:.2e}")
        
        # Save history
        history['loss'].append(avg_loss)
        history['bce'].append(avg_bce)
        history['dice'].append(avg_dice)
        history['focal'].append(avg_focal)
        history['dice_score'].append(avg_dice_score)
        history['iou'].append(avg_iou)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': config
            }, config['model_save_path'])
            print(f"      💾 New best model saved! (Loss: {best_loss:.4f})")
    
    print(f"\n🎉 Improved training completed!")
    print(f"   💾 Best model saved: {config['model_save_path']}")
    print(f"   🎯 Best training loss: {best_loss:.4f}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_improved_model()
