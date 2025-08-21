#!/usr/bin/env python3
"""
Local kidney detection training with proper numpy data
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

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
            print(f"   Sample {i}: {pair_info['source_file']} - {pair_info['mask_coverage']*100:.2f}% coverage")
    
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

def train_model(data_dir, epochs=50, lr=1e-4, device='cpu'):
    """Train the kidney detection model"""
    print(f"🚀 TRAINING KIDNEY DETECTION MODEL")
    print("=" * 50)
    print(f"📂 Data directory: {data_dir}")
    print(f"🔧 Device: {device}")
    print(f"📊 Epochs: {epochs}")
    print(f"📈 Learning rate: {lr}")
    
    # Create dataset
    dataset = KidneyDataset(data_dir)
    
    if len(dataset) == 0:
        print("❌ No training data found!")
        return None
    
    # Create data loader (no splitting since we have limited data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"📊 Training with {len(dataset)} samples")
    
    # Create model
    model = UNet3D(in_channels=1, out_channels=1, init_features=32)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    losses = []
    
    print(f"\n🎯 Starting training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for mri, mask in pbar:
                mri = mri.to(device)
                mask = mask.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(mri)
                loss = combined_loss(output, mask)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{epoch_loss/num_batches:.6f}'
                })
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = os.path.join(os.path.dirname(data_dir), 'kidney_model_local.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'metadata': dataset.metadata
            }, model_path)
            print(f"✅ Saved best model (loss: {best_loss:.6f}) to {model_path}")
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Best loss: {best_loss:.6f}")
    print(f"💾 Model saved to: {model_path}")
    
    # Plot training curve
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plot_path = os.path.join(os.path.dirname(data_dir), 'training_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Training curve saved to: {plot_path}")
    except:
        print("⚠️  Could not save training curve plot")
    
    return model_path

def main():
    """Main training function"""
    # Paths
    data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\training\kidneyTrainingData"
    
    # Check if training data exists
    if not os.path.exists(os.path.join(data_dir, 'metadata.pkl')):
        print("❌ No training data found! Run create_training_data.py first.")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Train model
    model_path = train_model(
        data_dir=data_dir,
        epochs=100,  # More epochs since we have limited data
        lr=1e-3,     # Higher learning rate for faster learning
        device=device
    )
    
    if model_path:
        print(f"✅ Training successful! Model saved to: {model_path}")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()
