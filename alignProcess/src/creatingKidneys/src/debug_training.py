#!/usr/bin/env python3
"""
Debug Kidney Detection Training
==============================

This script debugs the kidney detection training to understand why
the model is predicting everything as kidney.

Author: AI Assistant
Date: 2025-08-14
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import glob
import matplotlib.pyplot as plt

# Import model architecture
from unet_3d import UNet3D

def analyze_training_data():
    """Analyze the training data to understand the kidney/background ratio"""
    
    print("🔍 ANALYZING TRAINING DATA")
    print("="*50)
    
    data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    
    print(f"📁 Found {len(mat_files)} .mat files")
    
    total_samples = 0
    kidney_stats = []
    
    for mat_file in mat_files:
        print(f"\n📂 Processing: {os.path.basename(mat_file)}")
        
        # Load the .mat file
        try:
            data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
            continue
        
        if 'images' not in data:
            print(f"   ❌ No 'images' field found")
            continue
        
        images = data['images']
        if not hasattr(images, '__len__'):
            images = [images]
        
        # Find MRI images and their kidney slaves
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
            
            print(f"   🧠 Found MRI: {image_name} {shape}")
            
            # Look for kidney slaves
            kidney_mask = None
            kidney_slave_names = []
            
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
                        
                        # Check if this is a kidney slave
                        if "kidney" in slave_name.lower():
                            kidney_slave_names.append(slave_name)
                            print(f"      🫘 Found kidney slave: {slave_name}")
                            
                            if hasattr(slave, 'data') and slave.data is not None:
                                try:
                                    # Handle different data types
                                    slave_data = slave.data
                                    
                                    # Convert to numpy array if needed
                                    if hasattr(slave_data, 'astype'):
                                        slave_array = slave_data.astype(bool)
                                    else:
                                        slave_array = np.array(slave_data, dtype=bool)
                                    
                                    if kidney_mask is None:
                                        kidney_mask = slave_array
                                    else:
                                        kidney_mask |= slave_array
                                        
                                except Exception as e:
                                    print(f"      ⚠️  Could not process slave data: {e}")
                                    continue
            
            if kidney_mask is not None:
                total_samples += 1
                
                # Calculate statistics
                total_voxels = np.prod(kidney_mask.shape)
                kidney_voxels = np.sum(kidney_mask)
                background_voxels = total_voxels - kidney_voxels
                kidney_percent = kidney_voxels / total_voxels * 100
                
                print(f"      📊 Total voxels: {total_voxels:,}")
                print(f"      🫘 Kidney voxels: {kidney_voxels:,} ({kidney_percent:.2f}%)")
                print(f"      🌫️  Background voxels: {background_voxels:,} ({100-kidney_percent:.2f}%)")
                
                kidney_stats.append({
                    'file': os.path.basename(mat_file),
                    'image_name': image_name,
                    'shape': shape,
                    'total_voxels': total_voxels,
                    'kidney_voxels': kidney_voxels,
                    'kidney_percent': kidney_percent,
                    'kidney_slaves': kidney_slave_names
                })
            else:
                print(f"      ❌ No kidney slaves found")
    
    print(f"\n📊 TRAINING DATA SUMMARY")
    print("="*50)
    print(f"Total training samples: {total_samples}")
    
    if kidney_stats:
        kidney_percentages = [stat['kidney_percent'] for stat in kidney_stats]
        print(f"Kidney coverage range: {min(kidney_percentages):.2f}% - {max(kidney_percentages):.2f}%")
        print(f"Average kidney coverage: {np.mean(kidney_percentages):.2f}%")
        print(f"Median kidney coverage: {np.median(kidney_percentages):.2f}%")
        
        print(f"\n📋 Detailed breakdown:")
        for stat in kidney_stats:
            print(f"   {stat['file']} | {stat['image_name']} | {stat['kidney_percent']:.2f}% | {len(stat['kidney_slaves'])} slaves")
    
    return kidney_stats

def test_model_prediction():
    """Test current model prediction on a simple case"""
    
    print(f"\n🧠 TESTING CURRENT MODEL")
    print("="*50)
    
    model_path = "kidney_unet_model_best.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    device = torch.device('cpu')
    model = UNet3D(in_channels=1, out_channels=1)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create test inputs
    print("\n🧪 Testing with different inputs:")
    
    test_cases = [
        ("All zeros", np.zeros((64, 64, 32))),
        ("All ones", np.ones((64, 64, 32))),
        ("Random noise", np.random.random((64, 64, 32))),
        ("Center sphere", create_center_sphere(64, 64, 32, radius=10)),
        ("Edge pattern", create_edge_pattern(64, 64, 32))
    ]
    
    for name, test_input in test_cases:
        print(f"\n   🎯 Testing: {name}")
        
        # Normalize input
        test_normalized = (test_input - test_input.min()) / (test_input.max() - test_input.min() + 1e-8)
        input_tensor = torch.FloatTensor(test_normalized).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        print(f"      📊 Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
        print(f"      📈 Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")
        print(f"      📊 Mean prediction: {prediction.mean():.3f}")
        print(f"      🎯 Predictions > 0.5: {np.sum(prediction > 0.5)} voxels ({np.sum(prediction > 0.5)/np.prod(prediction.shape)*100:.1f}%)")

def create_center_sphere(x, y, z, radius=10):
    """Create a sphere in the center of the volume"""
    center_x, center_y, center_z = x//2, y//2, z//2
    sphere = np.zeros((x, y, z))
    
    for i in range(x):
        for j in range(y):
            for k in range(z):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2 + (k - center_z)**2)
                if dist <= radius:
                    sphere[i, j, k] = 1.0
    
    return sphere

def create_edge_pattern(x, y, z):
    """Create an edge pattern"""
    pattern = np.zeros((x, y, z))
    pattern[:5, :, :] = 1.0  # Top edge
    pattern[-5:, :, :] = 1.0  # Bottom edge
    pattern[:, :5, :] = 1.0  # Left edge
    pattern[:, -5:, :] = 1.0  # Right edge
    return pattern

if __name__ == "__main__":
    print("🔍 KIDNEY DETECTION DEBUG ANALYSIS")
    print("="*60)
    
    # Analyze training data
    kidney_stats = analyze_training_data()
    
    # Test current model
    test_model_prediction()
    
    print(f"\n✅ DEBUG ANALYSIS COMPLETE")
    print("="*60)
