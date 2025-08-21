#!/usr/bin/env python3
"""
Simple training with the 1 sample we have and then deploy it
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

def train_with_single_sample():
    """Train with the one sample we have"""
    print("🚀 TRAINING WITH SINGLE KIDNEY SAMPLE")
    print("=" * 50)
    
    data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\training\kidneyTrainingData"
    
    # Load the single training sample
    mri = np.load(os.path.join(data_dir, 'mri_000.npy')).astype(np.float32)
    mask = np.load(os.path.join(data_dir, 'mask_000.npy')).astype(np.float32)
    
    print(f"📊 MRI shape: {mri.shape}")
    print(f"📊 Mask shape: {mask.shape}")
    print(f"📊 Mask coverage: {np.sum(mask) / np.prod(mask.shape) * 100:.2f}%")
    
    # Add batch and channel dimensions
    mri_tensor = torch.FloatTensor(mri).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    mask_tensor = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    
    # Create model
    model = UNet3D(in_channels=1, out_channels=1, init_features=32)
    
    # Simple BCE loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for many epochs with the single sample
    model.train()
    best_loss = float('inf')
    
    print("\n🎯 Training with single sample...")
    
    for epoch in range(200):  # Many epochs to overfit to this sample
        optimizer.zero_grad()
        
        # Forward pass
        output = model(mri_tensor)
        loss = criterion(output, mask_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            model_path = os.path.join(os.path.dirname(data_dir), 'kidney_model_simple.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, model_path)
    
    print(f"\n✅ Training completed!")
    print(f"📊 Best loss: {best_loss:.6f}")
    print(f"💾 Model saved to: {model_path}")
    
    # Test the model on the training sample
    model.eval()
    with torch.no_grad():
        output = model(mri_tensor)
        prediction = torch.sigmoid(output).numpy()[0, 0]
        
        print(f"\n🧪 TEST ON TRAINING SAMPLE:")
        print(f"   Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")
        print(f"   Output mean: {prediction.mean():.3f}")
        print(f"   Pixels > 0.5: {np.sum(prediction > 0.5)} / {np.prod(prediction.shape)}")
    
    return model_path

def main():
    """Main function"""
    model_path = train_with_single_sample()
    return model_path

if __name__ == "__main__":
    main()
