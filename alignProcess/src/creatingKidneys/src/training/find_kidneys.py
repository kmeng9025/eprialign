#!/usr/bin/env python3
"""
Extract slave names and find kidney data from .mat files
"""

import os
import numpy as np
import scipy.io as sio

def extract_slave_name(slave):
    """Extract slave name from complex structure"""
    try:
        if hasattr(slave, 'dtype') and slave.dtype.names:
            # Try different name fields
            for name_field in ['name', 'Name', 'slave_name', 'slavename']:
                if name_field in slave.dtype.names:
                    name_data = slave[name_field][0, 0]
                    
                    # Handle different name data types
                    if isinstance(name_data, (str, np.str_)):
                        return str(name_data)
                    elif isinstance(name_data, np.ndarray):
                        if name_data.size > 0:
                            if name_data.dtype.kind in ['U', 'S']:  # Unicode or string
                                return str(name_data.flat[0])
                            elif name_data.dtype == object:
                                return str(name_data.flat[0])
    except:
        pass
    
    return "Unknown"

def extract_slave_mask(slave):
    """Extract slave mask from complex structure"""
    try:
        if hasattr(slave, 'dtype') and slave.dtype.names:
            # Try different mask fields
            for mask_field in ['mask', 'Mask', 'data', 'Data']:
                if mask_field in slave.dtype.names:
                    mask_data = slave[mask_field][0, 0]
                    
                    if isinstance(mask_data, np.ndarray) and mask_data.size > 0:
                        return mask_data
    except:
        pass
    
    return None

def find_detailed_slaves(mat_file):
    """Find all slaves in a mat file with detailed extraction"""
    print(f"\n🔍 DETAILED SLAVE EXTRACTION: {os.path.basename(mat_file)}")
    print("=" * 60)
    
    try:
        data = sio.loadmat(mat_file)
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")
        return []
    
    all_slaves = []
    
    # Look at images
    if 'images' in data:
        images = data['images']
        print(f"📸 Processing {images.shape[1]} images...")
        
        for i in range(images.shape[1]):
            try:
                img = images[0, i]
                
                # Get image name
                img_name = "Unknown"
                if hasattr(img, 'dtype') and img.dtype.names:
                    for name_field in ['Name', 'name', 'FileName']:
                        if name_field in img.dtype.names:
                            try:
                                name_data = img[name_field][0, 0]
                                if isinstance(name_data, np.ndarray) and name_data.size > 0:
                                    img_name = str(name_data.flat[0])
                                    break
                            except:
                                pass
                
                # Get image data shape
                img_shape = "Unknown"
                if hasattr(img, 'dtype') and img.dtype.names and 'data' in img.dtype.names:
                    try:
                        img_data = img['data'][0, 0]
                        if hasattr(img_data, 'shape'):
                            img_shape = img_data.shape
                    except:
                        pass
                
                print(f"\n   📷 Image {i}: '{img_name}' - shape: {img_shape}")
                
                # Look at slaves
                if hasattr(img, 'dtype') and img.dtype.names and 'slaves' in img.dtype.names:
                    slaves = img['slaves'][0, 0]
                    
                    if hasattr(slaves, 'shape') and slaves.size > 0:
                        print(f"      🔍 Found {slaves.shape[0]} slaves")
                        
                        for j in range(slaves.shape[0]):
                            try:
                                slave = slaves[j, 0]
                                
                                # Extract slave name
                                slave_name = extract_slave_name(slave)
                                
                                # Extract slave mask
                                slave_mask = extract_slave_mask(slave)
                                
                                mask_info = "No mask"
                                if slave_mask is not None:
                                    mask_info = f"mask shape: {slave_mask.shape}, dtype: {slave_mask.dtype}"
                                
                                print(f"         Slave {j}: '{slave_name}' - {mask_info}")
                                
                                # Check if this is a kidney
                                if 'kidney' in slave_name.lower():
                                    print(f"         🎯 KIDNEY FOUND: {slave_name}")
                                    
                                    # Store kidney data
                                    kidney_data = {
                                        'file': os.path.basename(mat_file),
                                        'image_index': i,
                                        'image_name': img_name,
                                        'image_shape': img_shape,
                                        'slave_index': j,
                                        'slave_name': slave_name,
                                        'mask': slave_mask
                                    }
                                    all_slaves.append(kidney_data)
                                
                                # Also check for other interesting slaves
                                if any(keyword in slave_name.lower() for keyword in ['roi', 'region', 'organ']):
                                    print(f"         🔍 Interesting slave: {slave_name}")
                                
                            except Exception as e:
                                print(f"         Error processing slave {j}: {e}")
                    else:
                        print(f"      No slaves found")
                
            except Exception as e:
                print(f"   Error processing image {i}: {e}")
    
    return all_slaves

def main():
    """Extract all kidney data from .mat files"""
    training_data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
    
    # Get all .mat files
    mat_files = []
    if os.path.exists(training_data_dir):
        for file in os.listdir(training_data_dir):
            if file.endswith('.mat'):
                mat_files.append(os.path.join(training_data_dir, file))
    
    print(f"🎯 KIDNEY EXTRACTION FROM {len(mat_files)} MAT FILES")
    print("=" * 70)
    
    all_kidneys = []
    
    # Process each file
    for mat_file in mat_files:
        kidneys = find_detailed_slaves(mat_file)
        all_kidneys.extend(kidneys)
    
    # Summary
    print(f"\n🎉 EXTRACTION SUMMARY")
    print("=" * 30)
    print(f"Total kidney slaves found: {len(all_kidneys)}")
    
    for i, kidney in enumerate(all_kidneys):
        print(f"\nKidney {i+1}:")
        print(f"   File: {kidney['file']}")
        print(f"   Image: {kidney['image_name']} (index {kidney['image_index']})")
        print(f"   Image shape: {kidney['image_shape']}")
        print(f"   Slave: {kidney['slave_name']} (index {kidney['slave_index']})")
        if kidney['mask'] is not None:
            print(f"   Mask: shape={kidney['mask'].shape}, dtype={kidney['mask'].dtype}")
            print(f"   Mask coverage: {np.sum(kidney['mask'] > 0) / np.prod(kidney['mask'].shape) * 100:.2f}%")
        else:
            print(f"   Mask: None")

if __name__ == "__main__":
    main()
