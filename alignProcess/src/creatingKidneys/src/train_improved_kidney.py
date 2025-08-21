"""
Improved AI Kidney Detection Training
====================================
A completely rewritten training pipeline with better data extraction and preprocessing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom, binary_erosion, binary_dilation, label
import matplotlib.pyplot as plt
from datetime import datetime
from unet_3d import UNet3D
import argparse

class ImprovedKidneyDataset(Dataset):
    """Improved dataset with better data extraction and augmentation"""
    
    def __init__(self, data_dir, debug=True):
        self.data_dir = data_dir
        self.debug = debug
        self.samples = []
        self.load_and_process_data()
    
    def load_and_process_data(self):
        """Load and intelligently process training data"""
        print("🔍 Loading and processing training data with improved extraction...")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.mat'):
                file_path = os.path.join(self.data_dir, filename)
                print(f"\n📁 Processing {filename}...")
                
                try:
                    data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
                    if 'images' not in data:
                        print(f"   ⚠️  No 'images' field in {filename}")
                        continue
                    
                    images = data['images']
                    if not hasattr(images, '__len__'):
                        print(f"   ⚠️  Invalid images structure in {filename}")
                        continue
                    
                    print(f"   📊 Found {len(images)} images")
                    
                    # Extract MRI and kidney data
                    self.extract_kidney_samples(images, filename)
                    
                except Exception as e:
                    print(f"   ❌ Error processing {filename}: {str(e)}")
                    continue
        
        print(f"\n📊 FINAL TRAINING DATA SUMMARY:")
        print(f"   Total samples: {len(self.samples)}")
        
        for i, sample in enumerate(self.samples):
            kidney_coverage = np.sum(sample['mask']) / np.prod(sample['mask'].shape) * 100
            print(f"   {i+1}. {sample['source']} | Kidney coverage: {kidney_coverage:.1f}% | Shape: {sample['mri'].shape}")
    
    def extract_kidney_samples(self, images, filename):
        """Extract MRI and kidney mask pairs with intelligent processing"""
        
        # First pass: identify all MRI images
        mri_images = []
        for i in range(len(images)):
            img = images[i]
            if not hasattr(img, 'data') or img.data is None:
                continue
                
            if not hasattr(img.data, 'shape') or len(img.data.shape) != 3:
                continue
            
            # Get image name
            name = self.get_image_name(img, i)
            shape = img.data.shape
            
            # Check if this looks like an MRI
            if self.is_mri_image(name, shape):
                mri_images.append({
                    'index': i,
                    'data': img.data,
                    'name': name,
                    'shape': shape,
                    'img_obj': img
                })
                print(f"   🧠 Found MRI: {name} {shape}")
        
        # Second pass: for each MRI, look for kidney annotations
        for mri_info in mri_images:
            self.extract_kidneys_from_mri(mri_info, filename)
    
    def get_image_name(self, img, index):
        """Extract image name safely"""
        if hasattr(img, 'Name') and img.Name is not None:
            if isinstance(img.Name, str):
                return img.Name
            elif hasattr(img.Name, '__len__'):
                try:
                    # Handle character arrays
                    if hasattr(img.Name, 'flatten'):
                        chars = img.Name.flatten()
                        return ''.join(chr(c) for c in chars if c != 0 and 32 <= c <= 126)
                    else:
                        return str(img.Name)
                except:
                    return f"img_{index}"
        return f"img_{index}"
    
    def is_mri_image(self, name, shape):
        """Determine if this is an MRI image"""
        name_lower = name.lower()
        
        # Check by name
        if 'mri' in name_lower:
            return True
        
        # Check by typical MRI dimensions
        if len(shape) == 3:
            # Common MRI sizes: 350x350xN, 256x256xN, 512x512xN
            if (shape[0] == shape[1] and shape[0] in [256, 350, 512] and 
                10 <= shape[2] <= 100):
                return True
        
        return False
    
    def extract_kidneys_from_mri(self, mri_info, filename):
        """Extract kidney masks from MRI image"""
        mri_data = mri_info['data'].astype(np.float32)
        mri_name = mri_info['name']
        img_obj = mri_info['img_obj']
        
        # Look for slaves (manual annotations) - NO synthetic generation
        kidney_masks = []
        if hasattr(img_obj, 'slaves') and img_obj.slaves is not None:
            slaves = img_obj.slaves
            if isinstance(slaves, np.ndarray):
                for slave in slaves.flatten():
                    if hasattr(slave, 'Name') and hasattr(slave, 'data'):
                        slave_name = self.get_image_name(slave, 0)
                        slave_name_lower = slave_name.lower()
                        
                        # Include ANY slave with "kidney" in name, but exclude "SRF"
                        if ('kidney' in slave_name_lower and 
                            'srf' not in slave_name_lower and
                            hasattr(slave.data, 'shape') and 
                            len(slave.data.shape) == 3 and
                            slave.data.shape == mri_data.shape):
                            
                            mask = (slave.data > 0).astype(np.float32)
                            kidney_coverage = np.sum(mask) / np.prod(mask.shape) * 100
                            
                            # Add ALL kidney slaves regardless of coverage
                            kidney_masks.append(mask)
                            print(f"      ✅ Found kidney slave: {slave_name} ({kidney_coverage:.1f}% coverage)")
        
        # Only add sample if we found actual kidney slaves (no synthetic)
        if len(kidney_masks) > 0:
            # Combine all kidney masks
            combined_mask = np.zeros_like(kidney_masks[0])
            for mask in kidney_masks:
                combined_mask = np.logical_or(combined_mask, mask > 0.5).astype(np.float32)
            
            # Add sample to dataset
            self.samples.append({
                'mri': mri_data,
                'mask': combined_mask,
                'source': f"{filename}:{mri_name}",
                'num_kidney_regions': len(kidney_masks)
            })
            
            coverage = np.sum(combined_mask) / np.prod(combined_mask.shape) * 100
            print(f"      ✅ Added training sample: {coverage:.1f}% kidney coverage")
        else:
            print(f"      ❌ No kidney slaves found for {mri_name} - skipping")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get MRI and mask
        mri = sample['mri'].copy()
        mask = sample['mask'].copy()
        
        # Improved normalization using robust statistics
        # Use percentiles to handle outliers better
        p1, p99 = np.percentile(mri[mri > 0], [1, 99])  # Exclude zero background
        if p99 > p1:
            mri = (mri - p1) / (p99 - p1)
            mri = np.clip(mri, 0, 1)
        else:
            # Fallback normalization
            mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
        
        # Resize to consistent training size
        target_shape = (64, 64, 32)
        zoom_factors = [target_shape[i] / mri.shape[i] for i in range(3)]
        
        mri_resized = zoom(mri, zoom_factors, order=1)
        mask_resized = zoom(mask, zoom_factors, order=0)  # Nearest neighbor for mask
        
        # Data augmentation (optional)
        if np.random.random() > 0.5:  # 50% chance
            mri_resized, mask_resized = self.augment_data(mri_resized, mask_resized)
        
        # Convert to tensors
        mri_tensor = torch.from_numpy(mri_resized).float().unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
        
        return mri_tensor, mask_tensor
    
    def augment_data(self, mri, mask):
        """Simple data augmentation"""
        # Random flip along one axis
        if np.random.random() > 0.5:
            axis = np.random.choice([0, 1, 2])
            mri = np.flip(mri, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        
        # Small rotation (simple shear)
        if np.random.random() > 0.7:
            # Add small noise to MRI (not mask)
            noise = np.random.normal(0, 0.02, mri.shape)
            mri = np.clip(mri + noise, 0, 1)
        
        return mri, mask

class ImprovedCombinedLoss(nn.Module):
    """Improved loss function for kidney segmentation"""
    
    def __init__(self, bce_weight=0.3, dice_weight=0.5, focal_weight=0.2):
        super(ImprovedCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))  # Give more weight to kidney pixels
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

def train_improved_model(data_dir=None):
    """Train an improved kidney detection model"""
    print("🚀 TRAINING IMPROVED AI KIDNEY DETECTION MODEL")
    print("=" * 80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")

    # Load dataset with improved extraction
    if data_dir is None:
        data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"

    print(f"📁 Using data_dir: {data_dir}")
    dataset = ImprovedKidneyDataset(data_dir, debug=True)

    if len(dataset) == 0:
        raise ValueError("No training data found!")

    # Create DataLoader
    batch_size = min(2, len(dataset))  # Don't exceed dataset size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,}")

    # Setup training
    criterion = ImprovedCombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

    # Training loop
    num_epochs = 150  # More epochs for better convergence
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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

        # Print progress more frequently
        if epoch % 10 == 0 or epoch == num_epochs - 1:
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
            
            # Save with timestamp
            model_path = f'kidney_improved_model_{timestamp}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'training_samples': len(dataset),
                'model_info': 'Improved kidney detection model with better data extraction'
            }, model_path)

            # Save as the new default model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'training_samples': len(dataset),
                'model_info': 'Improved kidney detection model with better data extraction'
            }, 'kidney_unet_model_improved.pth')

            print(f"   💾 New best model saved! Loss: {best_loss:.4f}")

    print(f"\n✅ Training completed!")
    print(f"📊 Final best loss: {best_loss:.4f}")
    print(f"🎯 Model saved as: kidney_unet_model_improved.pth")
    print(f"📈 Training samples used: {len(dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved kidney detection model")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to training data directory')
    args = parser.parse_args()
    train_improved_model(data_dir=args.data_dir)
