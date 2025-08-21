#!/usr/bin/env python3
"""
Deep inspection of slaves array structure
"""

import os
import numpy as np
import scipy.io as sio

def extract_slave_name(slave):
    """Extract slave name from complex structure"""
    try:
        if hasattr(slave, 'dtype') and slave.dtype.names:
            for name_field in ['name', 'Name', 'slave_name', 'slavename']:
                if name_field in slave.dtype.names:
                    name_data = slave[name_field][0, 0]
                    
                    if isinstance(name_data, (str, np.str_)):
                        return str(name_data)
                    elif isinstance(name_data, np.ndarray):
                        if name_data.size > 0:
                            if name_data.dtype.kind in ['U', 'S']:
                                return str(name_data.flat[0])
                            elif name_data.dtype == object:
                                return str(name_data.flat[0])
    except:
        pass
    return "Unknown"

def deep_check_slaves(mat_file_path):
    """Deep check of slaves structure"""
    print(f"\n🔍 Deep checking: {os.path.basename(mat_file_path)}")
    
    try:
        data = sio.loadmat(mat_file_path)
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        return
    
    if 'images' in data:
        images = data['images']
        
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
                
                # Check if it's an MRI image
                img_name_lower = img_name.lower()
                is_mri = "mri" in img_name_lower
                has_long = "long" in img_name_lower
                
                if is_mri and not has_long:
                    print(f"   📷 MRI Image {i}: '{img_name}'")
                    
                    # Deep inspect slaves structure
                    if hasattr(img, 'dtype') and img.dtype.names and 'slaves' in img.dtype.names:
                        slaves = img['slaves'][0, 0]
                        
                        print(f"      🔍 Slaves array shape: {getattr(slaves, 'shape', 'N/A')}")
                        print(f"      🔍 Slaves array type: {type(slaves)}")
                        
                        if hasattr(slaves, 'shape') and slaves.size > 0:
                            # Check all dimensions of slaves array
                            print(f"      🔍 Exploring all slaves in {slaves.shape}...")
                            
                            # Try different indexing patterns
                            if len(slaves.shape) == 2:  # (rows, cols)
                                for row in range(slaves.shape[0]):
                                    for col in range(slaves.shape[1]):
                                        try:
                                            slave = slaves[row, col]
                                            if slave.size > 0:  # Only process non-empty slaves
                                                slave_name = extract_slave_name(slave)
                                                
                                                # Check if it's a kidney
                                                is_kidney = 'kidney' in slave_name.lower()
                                                has_srf = 'srf' in slave_name.lower()
                                                
                                                kidney_indicator = "🎯 KIDNEY" if is_kidney and not has_srf else ""
                                                srf_indicator = "❌ SRF" if has_srf else ""
                                                
                                                print(f"         Slave [{row},{col}]: '{slave_name}' {kidney_indicator} {srf_indicator}")
                                        except Exception as e:
                                            if slaves[row, col].size > 0:  # Only show errors for non-empty
                                                print(f"         Slave [{row},{col}]: Error - {e}")
                            else:
                                print(f"      Unexpected slaves shape: {slaves.shape}")
                        else:
                            print(f"      No slaves found or empty array")
                
            except Exception as e:
                print(f"   ❌ Error processing image {i}: {e}")

def main():
    """Check files with deep inspection"""
    training_data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
    
    # Get all .mat files
    mat_files = []
    if os.path.exists(training_data_dir):
        for file in os.listdir(training_data_dir):
            if file.endswith('.mat'):
                mat_files.append(os.path.join(training_data_dir, file))
    
    print(f"🎯 DEEP CHECKING MRI SLAVES IN {len(mat_files)} FILES")
    print("=" * 70)
    
    # Focus on a few files that likely have kidney data
    focus_files = ['HemoB6M022_better.mat', 'HemoB6M024.mat', 'HemoM003.mat', 'HemoM004.mat']
    
    for mat_file in mat_files:
        if any(focus in mat_file for focus in focus_files):
            deep_check_slaves(mat_file)

if __name__ == "__main__":
    main()
