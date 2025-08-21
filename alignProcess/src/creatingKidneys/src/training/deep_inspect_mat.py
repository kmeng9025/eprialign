#!/usr/bin/env python3
"""
Deep inspection of .mat file structure for kidney data
"""

import os
import numpy as np
import scipy.io as sio

def deep_inspect_mat(filepath):
    """Deep inspection of .mat file structure"""
    print(f"\n{'='*60}")
    print(f"DEEP INSPECTION: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        data = sio.loadmat(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return
    
    # Look at images
    if 'images' in data:
        images = data['images']
        print(f"\n📸 IMAGES: shape={images.shape}")
        
        if images.size > 0:
            for i in range(min(images.shape[1], 10)):  # Look at first 10 images
                try:
                    img = images[0, i]
                    print(f"   Image {i}: type={type(img)}")
                    
                    if hasattr(img, 'dtype') and img.dtype.names:
                        print(f"      Fields: {img.dtype.names}")
                        
                        # Look for name and data
                        if 'name' in img.dtype.names:
                            name = img['name'][0, 0]
                            if hasattr(name, 'shape') and name.size > 0:
                                name_str = str(name[0]) if name.ndim > 0 else str(name)
                                print(f"      Name: '{name_str}'")
                        
                        if 'data' in img.dtype.names:
                            img_data = img['data'][0, 0]
                            print(f"      Data shape: {getattr(img_data, 'shape', 'N/A')}")
                            print(f"      Data type: {getattr(img_data, 'dtype', 'N/A')}")
                        
                        # Look for slaves in image
                        if 'slaves' in img.dtype.names:
                            slaves = img['slaves'][0, 0]
                            print(f"      Slaves: shape={getattr(slaves, 'shape', 'N/A')}")
                            
                            if hasattr(slaves, 'shape') and slaves.size > 0:
                                print(f"      Found {slaves.shape[0]} slaves")
                                
                                # Look at each slave
                                for j in range(min(slaves.shape[0], 5)):  # First 5 slaves
                                    try:
                                        slave = slaves[j, 0]
                                        if hasattr(slave, 'dtype') and slave.dtype.names:
                                            slave_name = "Unknown"
                                            if 'name' in slave.dtype.names:
                                                slave_name_data = slave['name'][0, 0]
                                                if hasattr(slave_name_data, 'shape') and slave_name_data.size > 0:
                                                    slave_name = str(slave_name_data[0]) if slave_name_data.ndim > 0 else str(slave_name_data)
                                            
                                            mask_info = "No mask"
                                            if 'mask' in slave.dtype.names:
                                                mask = slave['mask'][0, 0]
                                                if hasattr(mask, 'shape'):
                                                    mask_info = f"mask shape={mask.shape}"
                                            
                                            print(f"         Slave {j}: '{slave_name}' - {mask_info}")
                                            
                                            # Check if this is a kidney
                                            if 'kidney' in slave_name.lower():
                                                print(f"         🎯 KIDNEY FOUND: {slave_name}")
                                    except Exception as e:
                                        print(f"         Error inspecting slave {j}: {e}")
                        
                except Exception as e:
                    print(f"   Error inspecting image {i}: {e}")
    
    # Look at sequences
    if 'sequences' in data:
        sequences = data['sequences']
        print(f"\n🎬 SEQUENCES: shape={sequences.shape}")
        
        if sequences.size > 0:
            for i in range(min(sequences.shape[1], 5)):  # Look at first 5 sequences
                try:
                    seq = sequences[0, i]
                    print(f"   Sequence {i}: type={type(seq)}")
                    
                    if hasattr(seq, 'dtype') and seq.dtype.names:
                        print(f"      Fields: {seq.dtype.names}")
                        
                        if 'name' in seq.dtype.names:
                            name = seq['name'][0, 0]
                            if hasattr(name, 'shape') and name.size > 0:
                                name_str = str(name[0]) if name.ndim > 0 else str(name)
                                print(f"      Name: '{name_str}'")
                        
                        # Look for slaves in sequence
                        if 'slaves' in seq.dtype.names:
                            slaves = seq['slaves'][0, 0]
                            print(f"      Slaves: shape={getattr(slaves, 'shape', 'N/A')}")
                            
                            if hasattr(slaves, 'shape') and slaves.size > 0:
                                print(f"      Found {slaves.shape[0]} slaves in sequence")
                                
                                # Look at each slave
                                for j in range(min(slaves.shape[0], 5)):  # First 5 slaves
                                    try:
                                        slave = slaves[j, 0]
                                        if hasattr(slave, 'dtype') and slave.dtype.names:
                                            slave_name = "Unknown"
                                            if 'name' in slave.dtype.names:
                                                slave_name_data = slave['name'][0, 0]
                                                if hasattr(slave_name_data, 'shape') and slave_name_data.size > 0:
                                                    slave_name = str(slave_name_data[0]) if slave_name_data.ndim > 0 else str(slave_name_data)
                                            
                                            print(f"         Sequence slave {j}: '{slave_name}'")
                                            
                                            # Check if this is a kidney
                                            if 'kidney' in slave_name.lower():
                                                print(f"         🎯 KIDNEY FOUND in sequence: {slave_name}")
                                    except Exception as e:
                                        print(f"         Error inspecting sequence slave {j}: {e}")
                
                except Exception as e:
                    print(f"   Error inspecting sequence {i}: {e}")

def main():
    """Deep inspect .mat files"""
    training_data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
    
    # Get all .mat files
    mat_files = []
    if os.path.exists(training_data_dir):
        for file in os.listdir(training_data_dir):
            if file.endswith('.mat'):
                mat_files.append(os.path.join(training_data_dir, file))
    
    # Deep inspect first 3 files
    for mat_file in mat_files[:3]:
        deep_inspect_mat(mat_file)

if __name__ == "__main__":
    main()
