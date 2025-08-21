#!/usr/bin/env python3
"""
Kidney Detection Model Training - All Data Version
================================================

This script trains a U-Net model for kidney detection using ALL available
training data without validation split, as requested by the user.

Features:
- Uses all training data for training (no validation split)
- Supports multiple MRI images per .mat file
- Data augmentation for better generalization
- Progressive learning rate scheduling
- Saves best model based on training loss

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

class KidneyDataset(Dataset):
    """Dataset for kidney segmentation training"""
    
    def __init__(self, data_dir, augment=True):
        self.data_dir = data_dir
        self.augment = augment
        self.samples = []
        
        print(f"🔍 Loading training data from: {data_dir}")
        
        # Find all .mat files in training directory
        mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
        print(f"   📁 Found {len(mat_files)} .mat files")
        
        # Process each file to extract MRI-kidney pairs
        for mat_file in mat_files:
            try:
                self._process_mat_file(mat_file)
            except Exception as e:
                print(f"   ⚠️  Warning: Could not process {os.path.basename(mat_file)}: {e}")
        
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
                                        # Already numpy array
                                        slave_array = slave_data.astype(bool)
                                    else:
                                        # Convert mat_struct or other types to numpy
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
                print(f"         ✅ Kidney mask: {kidney_count} voxels ({kidney_percent:.2f}%)")
                
                # Ensure MRI data is also properly converted
                try:
                    mri_array = np.array(img.data, dtype=np.float32)
                    
                    # Add this sample
                    self.samples.append({
                        'mri_data': mri_array,
                        'kidney_mask': kidney_mask.astype(np.float32),
                        'source_file': os.path.basename(mat_file),
                        'image_name': image_name
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
        
        # Normalize MRI data
        mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min() + 1e-8)
        
        # Resize to target size for U-Net
        target_size = (64, 64, 32)
        
        # Simple resize (more sophisticated augmentation could be added)
        from scipy.ndimage import zoom
        
        zoom_factors = [t/s for t, s in zip(target_size, mri_data.shape)]
        mri_resized = zoom(mri_data, zoom_factors, order=1)
        mask_resized = zoom(kidney_mask, zoom_factors, order=0)  # Nearest neighbor for masks
        
        # Data augmentation (if enabled)
        if self.augment:
            # Random rotation (simple implementation)
            if np.random.random() > 0.5:
                # Flip along x-axis
                mri_resized = mri_resized[::-1, :, :].copy()
                mask_resized = mask_resized[::-1, :, :].copy()
            
            if np.random.random() > 0.5:
                # Flip along y-axis
                mri_resized = mri_resized[:, ::-1, :].copy()
                mask_resized = mask_resized[:, ::-1, :].copy()
            
            # Random intensity scaling
            intensity_factor = np.random.uniform(0.8, 1.2)
            mri_resized = np.clip(mri_resized * intensity_factor, 0, 1)
        
        # Ensure arrays are contiguous
        mri_resized = np.ascontiguousarray(mri_resized)
        mask_resized = np.ascontiguousarray(mask_resized)
        
        # Convert to tensors
        mri_tensor = torch.FloatTensor(mri_resized).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.FloatTensor(mask_resized).unsqueeze(0)
        
        return mri_tensor, mask_tensor

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2 * intersection + self.smooth) / (predictions_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss"""
    
    def __init__(self, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice

def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate evaluation metrics"""
    predictions = torch.sigmoid(predictions)
    pred_binary = (predictions > threshold).float()
    
    # Flatten for metrics calculation
    pred_flat = pred_binary.cpu().numpy().flatten()
    target_flat = targets.cpu().numpy().flatten()
    
    # Dice coefficient
    intersection = np.sum(pred_flat * target_flat)
    dice = (2 * intersection) / (np.sum(pred_flat) + np.sum(target_flat) + 1e-8)
    
    # IoU (Jaccard index)
    iou = jaccard_score(target_flat, pred_flat, average='binary', zero_division=0)
    
    return dice, iou

def train_model():
    """Main training function"""
    
    print("🚀 KIDNEY DETECTION MODEL TRAINING")
    print("="*60)
    print("📋 Configuration: ALL DATA FOR TRAINING (No validation split)")
    
    # Configuration
    config = {
        'data_dir': r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training",
        'model_save_path': r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\kidney_unet_model_best.pth",
        'batch_size': 2,  # Small batch size due to memory constraints
        'num_epochs': 20,  # Reduced for faster testing
        'learning_rate': 1e-4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"📂 Training data directory: {config['data_dir']}")
    print(f"💾 Model save path: {config['model_save_path']}")
    print(f"🔧 Device: {config['device']}")
    print(f"📊 Batch size: {config['batch_size']}")
    print(f"🔄 Epochs: {config['num_epochs']}")
    print(f"📈 Learning rate: {config['learning_rate']}")
    
    # Create dataset (ALL data for training)
    print("\n📚 Creating training dataset...")
    train_dataset = KidneyDataset(config['data_dir'], augment=True)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )
    
    print(f"   ✅ Training batches: {len(train_loader)}")
    
    # Initialize model
    print("\n🧠 Initializing U-Net model...")
    model = UNet3D(in_channels=1, out_channels=1)
    model = model.to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   🔧 Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.3)  # Emphasize Dice loss more
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {config['num_epochs']} epochs...")
    print("="*60)
    
    best_loss = float('inf')
    training_history = {'loss': [], 'dice': [], 'iou': []}
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        
        print(f"\n📅 Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, (mri_data, kidney_masks) in enumerate(train_loader):
            # Move data to device
            mri_data = mri_data.to(config['device'])
            kidney_masks = kidney_masks.to(config['device'])
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(mri_data)
            loss = criterion(predictions, kidney_masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            dice, iou = calculate_metrics(predictions, kidney_masks)
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou
            
            # Print progress
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                print(f"   📊 Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, Dice={dice:.4f}, IoU={iou:.4f}")
        
        # Calculate epoch averages
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Record history
        training_history['loss'].append(avg_loss)
        training_history['dice'].append(avg_dice)
        training_history['iou'].append(avg_iou)
        
        print(f"   🎯 Epoch {epoch+1} Results:")
        print(f"      Loss: {avg_loss:.4f}")
        print(f"      Dice: {avg_dice:.4f}")
        print(f"      IoU:  {avg_iou:.4f}")
        print(f"      LR:   {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'training_history': training_history,
                'config': config
            }
            
            torch.save(checkpoint, config['model_save_path'])
            print(f"      💾 New best model saved! (Loss: {best_loss:.4f})")
        
        # Early stopping check
        if epoch > 20 and avg_loss > training_history['loss'][-10]:
            print(f"      📈 Loss not improving, but continuing as requested...")
    
    print(f"\n🎉 Training completed!")
    print(f"   💾 Best model saved: {config['model_save_path']}")
    print(f"   🎯 Best training loss: {best_loss:.4f}")
    
    # Plot training history
    print(f"\n📊 Plotting training history...")
    
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(training_history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Dice plot
    plt.subplot(1, 3, 2)
    plt.plot(training_history['dice'])
    plt.title('Training Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.grid(True)
    
    # IoU plot
    plt.subplot(1, 3, 3)
    plt.plot(training_history['iou'])
    plt.title('Training IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = config['model_save_path'].replace('.pth', '_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   📈 Training history plot saved: {plot_path}")
    
    plt.show()
    
    return model, training_history

if __name__ == "__main__":
    print("🤖 AI KIDNEY DETECTION TRAINING")
    print("Using ALL available training data (no validation split)")
    print("="*60)
    
    try:
        # Train the model
        model, history = train_model()
        
        print(f"\n✅ SUCCESS! Kidney detection model training complete!")
        print(f"🎯 Model ready for deployment in ai_kidney_detection.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise
