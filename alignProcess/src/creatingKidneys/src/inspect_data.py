#!/usr/bin/env python3
"""
Inspect Training Data Structure
===============================

This script inspects the actual content of kidney slaves to understand
why they're covering 100% of the volume.

Author: AI Assistant  
Date: 2025-08-14
"""

import scipy.io as sio
import numpy as np

def inspect_kidney_slave_data():
    """Inspect the actual kidney slave data structure"""
    
    print("🔍 INSPECTING KIDNEY SLAVE DATA")
    print("="*50)
    
    # Load one training file
    file_path = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\ExchangeB6M005.mat"
    print(f"📂 Loading: {file_path}")
    
    try:
        data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        print(f"✅ File loaded successfully")
        
        if 'images' not in data:
            print("❌ No 'images' field found")
            return
        
        images = data['images']
        if not hasattr(images, '__len__'):
            images = [images]
        
        print(f"📊 Found {len(images)} images")
        
        # Look at first MRI image
        for i, img in enumerate(images):
            if not hasattr(img, 'data') or img.data is None:
                continue
                
            # Get image name
            image_name = ""
            if hasattr(img, 'Name') and img.Name is not None:
                if isinstance(img.Name, str):
                    image_name = img.Name
                elif hasattr(img.Name, '__len__'):
                    try:
                        image_name = ''.join(chr(c) for c in img.Name.flatten() if c != 0)
                    except:
                        image_name = str(img.Name)
            
            # Check if it's an MRI
            shape = img.data.shape if hasattr(img.data, 'shape') else None
            if shape and len(shape) == 3 and "mri" in image_name.lower():
                print(f"\n🧠 Analyzing MRI: {image_name} {shape}")
                
                # Analyze MRI data
                mri_data = np.array(img.data)
                print(f"   📊 MRI data type: {mri_data.dtype}")
                print(f"   📊 MRI range: [{mri_data.min()}, {mri_data.max()}]")
                print(f"   📊 MRI mean: {mri_data.mean():.3f}")
                print(f"   📊 Non-zero voxels: {np.sum(mri_data != 0):,} ({np.sum(mri_data != 0)/np.prod(mri_data.shape)*100:.2f}%)")
                
                # Analyze kidney slaves
                if hasattr(img, 'slaves') and img.slaves is not None:
                    if not hasattr(img.slaves, '__len__'):
                        slaves = [img.slaves]
                    else:
                        slaves = img.slaves
                    
                    print(f"   🫘 Found {len(slaves)} slaves")
                    
                    for j, slave in enumerate(slaves):
                        if hasattr(slave, 'Name') and slave.Name is not None:
                            slave_name = ""
                            if isinstance(slave.Name, str):
                                slave_name = slave.Name
                            elif hasattr(slave.Name, '__len__'):
                                try:
                                    slave_name = ''.join(chr(c) for c in slave.Name.flatten() if c != 0)
                                except:
                                    slave_name = str(slave.Name)
                            
                            print(f"\n   🔸 Slave {j+1}: {slave_name}")
                            
                            if "kidney" in slave_name.lower():
                                print(f"      🫘 This is a KIDNEY slave")
                                
                                if hasattr(slave, 'data') and slave.data is not None:
                                    print(f"      📊 Slave data type: {type(slave.data)}")
                                    print(f"      📊 Slave data shape: {slave.data.shape if hasattr(slave.data, 'shape') else 'No shape'}")
                                    
                                    try:
                                        # Convert to numpy and analyze
                                        if hasattr(slave.data, 'astype'):
                                            slave_array = slave.data
                                        else:
                                            slave_array = np.array(slave.data)
                                        
                                        print(f"      📊 Converted type: {slave_array.dtype}")
                                        print(f"      📊 Array shape: {slave_array.shape}")
                                        print(f"      📊 Unique values: {np.unique(slave_array)}")
                                        print(f"      📊 Value counts:")
                                        unique_vals, counts = np.unique(slave_array, return_counts=True)
                                        for val, count in zip(unique_vals, counts):
                                            percent = count / np.prod(slave_array.shape) * 100
                                            print(f"         Value {val}: {count:,} voxels ({percent:.2f}%)")
                                        
                                        # Check if this is binary data
                                        if len(unique_vals) <= 2:
                                            print(f"      ✅ This appears to be binary mask data")
                                            if np.any(slave_array == 0):
                                                mask = slave_array.astype(bool)
                                                kidney_voxels = np.sum(mask)
                                                total_voxels = np.prod(mask.shape)
                                                print(f"      🎯 ACTUAL kidney coverage: {kidney_voxels:,}/{total_voxels:,} ({kidney_voxels/total_voxels*100:.2f}%)")
                                            else:
                                                print(f"      ⚠️  No zero values found - might not be a proper mask")
                                        else:
                                            print(f"      ⚠️  More than 2 unique values - not binary")
                                            
                                    except Exception as e:
                                        print(f"      ❌ Error analyzing slave data: {e}")
                                else:
                                    print(f"      ❌ No slave data found")
                            else:
                                print(f"      🔸 This is NOT a kidney slave")
                
                break  # Only analyze first MRI
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_kidney_slave_data()
