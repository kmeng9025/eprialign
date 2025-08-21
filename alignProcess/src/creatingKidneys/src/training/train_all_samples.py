#!/usr/bin/env python3
"""
Train kidney detection model with all 7 samples
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

# Add the parent directory to path to import UNet3D
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet_3d import UNet3D

class KidneyDataset(Dataset):
    """Dataset for loading kidney MRI and mask pairs"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.total_pairs = self.metadata['total_pairs']
        print(f"📂 Loaded dataset with {self.total_pairs} pairs")
        
        # Print sample information
        for i, pair_info in enumerate(self.metadata['pairs']):
            print(f"   Sample {i}: {pair_info['source_file']} - {pair_info['slave_name']} - {pair_info['mask_coverage']*100:.2f}% coverage")
    
    def __len__(self):
        return self.total_pairs
    
    def __getitem__(self, idx):
        # Load MRI and mask
        mri_path = os.path.join(self.data_dir, f'mri_{idx:03d}.npy')
        mask_path = os.path.join(self.data_dir, f'mask_{idx:03d}.npy')
        
        mri = np.load(mri_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        
        # Add channel dimension
        mri = mri[np.newaxis, ...]  # (1, H, W, D)
        mask = mask[np.newaxis, ...]  # (1, H, W, D)
        
        return torch.from_numpy(mri), torch.from_numpy(mask)

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation"""
    pred = torch.sigmoid(pred)
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def combined_loss(pred, target):
    """Combined BCE + Dice loss"""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice

def train_model():
    """Train the kidney detection model with all samples"""
    print("🚀 TRAINING KIDNEY DETECTION MODEL WITH ALL SAMPLES")
    print("=" * 60)
    
    data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\training\kidneyTrainingData"
    
    # Create dataset
    dataset = KidneyDataset(data_dir)
    
    if len(dataset) == 0:
        print("❌ No training data found!")
        return None
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"📊 Training with {len(dataset)} samples")
    
    # Create model
    model = UNet3D(in_channels=1, out_channels=1, init_features=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    print(f"\n🎯 Starting training...")
    
    for epoch in range(300):  # Many epochs since we have limited data
        epoch_loss = 0.0
        num_batches = 0
        
        for mri, mask in dataloader:
            # Forward pass
            optimizer.zero_grad()
            output = model(mri)
            loss = combined_loss(output, mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}/300 | Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(os.path.dirname(data_dir), 'kidney_model_final.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'metadata': dataset.metadata
            }, model_path)
            if epoch % 20 == 0:
                print(f"   ✅ Saved best model (loss: {best_loss:.6f})")
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Best loss: {best_loss:.6f}")
    print(f"💾 Model saved to: {model_path}")
    
    # Test the model on all training samples
    model.eval()
    print(f"\n🧪 TESTING ON ALL TRAINING SAMPLES:")
    
    with torch.no_grad():
        for i, (mri, mask) in enumerate(dataloader):
            output = model(mri)
            prediction = torch.sigmoid(output).numpy()[0, 0]
            target = mask.numpy()[0, 0]
            
            pred_pixels = np.sum(prediction > 0.5)
            target_pixels = np.sum(target > 0.5)
            
            print(f"   Sample {i}: pred_pixels={pred_pixels}, target_pixels={target_pixels}, ratio={pred_pixels/max(target_pixels,1):.2f}")
    
    return model_path

def main():
    """Main function"""
    model_path = train_model()
    return model_path

if __name__ == "__main__":
    main()
