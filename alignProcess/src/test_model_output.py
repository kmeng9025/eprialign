"""
Quick test to see what the model is actually outputting
"""
import torch
import numpy as np
import scipy.io as sio
from src.models.unet3d import UNet3D

# Load model
device = torch.device('cpu')
model = UNet3D(in_channels=1, out_channels=1)
checkpoint = torch.load('kidney_unet_model_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("📊 Model loaded, checking predictions...")

# Load a real MRI to test
data = sio.loadmat('data/withoutROIwithMRI.mat')
mri_data = data['>MRI']
print(f"📏 MRI shape: {mri_data.shape}")
print(f"📊 MRI range: [{mri_data.min():.3f}, {mri_data.max():.3f}]")

# Normalize like in training
mri_norm = (mri_data - np.percentile(mri_data, 1)) / (np.percentile(mri_data, 99) - np.percentile(mri_data, 1))
mri_norm = np.clip(mri_norm, 0, 1)

# Resize to model input size
from scipy.ndimage import zoom
target_shape = (64, 64, 32)
zoom_factors = [target_shape[i] / mri_data.shape[i] for i in range(3)]
mri_resized = zoom(mri_norm, zoom_factors, order=1)

print(f"📏 Resized shape: {mri_resized.shape}")
print(f"📊 Normalized range: [{mri_resized.min():.3f}, {mri_resized.max():.3f}]")

# Convert to tensor
input_tensor = torch.from_numpy(mri_resized).float().unsqueeze(0).unsqueeze(0)
print(f"🎯 Input tensor shape: {input_tensor.shape}")

# Run prediction
with torch.no_grad():
    pred = model(input_tensor)
    pred = torch.sigmoid(pred)
    
pred_np = pred.squeeze().numpy()
print(f"📈 Prediction shape: {pred_np.shape}")
print(f"📈 Prediction range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
print(f"📊 Prediction mean: {pred_np.mean():.4f}")
print(f"📊 Prediction std: {pred_np.std():.4f}")

# Check different threshold levels
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for thresh in thresholds:
    positive_pixels = (pred_np > thresh).sum()
    percentage = (positive_pixels / pred_np.size) * 100
    print(f"🎯 Threshold {thresh}: {positive_pixels} pixels ({percentage:.2f}%)")

# Check if all predictions are similar (model collapsed)
unique_vals = np.unique(pred_np)
print(f"🔍 Unique prediction values: {len(unique_vals)}")
print(f"🔍 First 10 unique values: {unique_vals[:10]}")
if len(unique_vals) < 100:
    print("⚠️  WARNING: Very few unique prediction values - model may have collapsed!")
