#!/usr/bin/env python3
"""
Check what slaves exist in MRI images
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

def check_mri_slaves(mat_file_path):
    """Check what slaves exist in MRI images"""
    print(f"\n🔍 Checking: {os.path.basename(mat_file_path)}")
    
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
                
                # Check if it's an MRI image (case insensitive)
                img_name_lower = img_name.lower()
                is_mri = "mri" in img_name_lower
                has_long = "long" in img_name_lower
                
                if is_mri and not has_long:
                    print(f"   📷 MRI Image {i}: '{img_name}'")
                    
                    # Look at slaves
                    if hasattr(img, 'dtype') and img.dtype.names and 'slaves' in img.dtype.names:
                        slaves = img['slaves'][0, 0]
                        
                        if hasattr(slaves, 'shape') and slaves.size > 0:
                            print(f"      🔍 Found {slaves.shape[0]} slaves:")
                            
                            # Show ALL slaves, not just the first one
                            for j in range(slaves.shape[0]):
                                try:
                                    slave = slaves[j, 0]
                                    slave_name = extract_slave_name(slave)
                                    
                                    # Check if it's a kidney
                                    is_kidney = 'kidney' in slave_name.lower()
                                    has_srf = 'srf' in slave_name.lower()
                                    
                                    kidney_indicator = "🎯 KIDNEY" if is_kidney and not has_srf else ""
                                    srf_indicator = "❌ SRF" if has_srf else ""
                                    
                                    print(f"         Slave {j}: '{slave_name}' {kidney_indicator} {srf_indicator}")
                                except Exception as e:
                                    print(f"         Slave {j}: Error - {e}")
                        else:
                            print(f"      No slaves found")
                
            except Exception as e:
                print(f"   ❌ Error processing image {i}: {e}")

def main():
    """Check all files"""
    training_data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
    
    # Get all .mat files
    mat_files = []
    if os.path.exists(training_data_dir):
        for file in os.listdir(training_data_dir):
            if file.endswith('.mat'):
                mat_files.append(os.path.join(training_data_dir, file))
    
    print(f"🎯 CHECKING MRI SLAVES IN {len(mat_files)} FILES")
    print("=" * 60)
    
    for mat_file in mat_files:
        check_mri_slaves(mat_file)

if __name__ == "__main__":
    main()
