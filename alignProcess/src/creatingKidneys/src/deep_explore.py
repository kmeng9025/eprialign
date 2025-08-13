"""
Deep exploration of training data images to find MRI and kidney annotations
"""
import numpy as np
import scipy.io
import os
from pathlib import Path

def explore_images(filepath):
    """Explore images in a .mat file"""
    print(f"\n{'='*50}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    try:
        data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        
        if 'images' in data:
            images = data['images']
            print(f"Images array shape: {images.shape}")
            
            if len(images.shape) == 2:
                for i in range(images.shape[1]):  # iterate over columns
                    img = images[0, i]  # get image from first row
                    
                    if img is None or (isinstance(img, np.ndarray) and img.size == 0):
                        print(f"  Image {i}: Empty")
                        continue
                    
                    print(f"  Image {i}:")
                    
                    # Check for common attributes
                    if hasattr(img, 'type'):
                        print(f"    Type: {img.type}")
                    if hasattr(img, 'name'):
                        print(f"    Name: {img.name}")
                    if hasattr(img, 'filename'):
                        print(f"    Filename: {img.filename}")
                    
                    # Look for data
                    if hasattr(img, 'data') and img.data is not None:
                        print(f"    Has data: {type(img.data)}")
                        
                        # Check for MRI-like data
                        if hasattr(img.data, 'shape'):
                            print(f"    Data shape: {img.data.shape}")
                            
                            # Check if this looks like MRI data (3D)
                            if len(img.data.shape) == 3:
                                print(f"    *** POTENTIAL MRI DATA (3D) ***")
                        
                        # If it's a structured array, check fields
                        if hasattr(img.data, 'dtype') and img.data.dtype.names:
                            print(f"    Data fields: {img.data.dtype.names}")
                            
                            for field in img.data.dtype.names:
                                if 'MRI' in field or 'mri' in field:
                                    print(f"    *** MRI FIELD FOUND: {field} ***")
                    
                    # Look for slaves (annotations)
                    if hasattr(img, 'slaves') and img.slaves is not None:
                        if isinstance(img.slaves, np.ndarray) and img.slaves.size > 0:
                            print(f"    Slaves: {img.slaves.shape}")
                            
                            if len(img.slaves.shape) == 2:
                                for j in range(img.slaves.shape[1]):
                                    slave = img.slaves[0, j]
                                    
                                    if slave is None or (isinstance(slave, np.ndarray) and slave.size == 0):
                                        continue
                                    
                                    slave_name = getattr(slave, 'name', 'Unnamed')
                                    slave_type = getattr(slave, 'type', 'Unknown')
                                    
                                    print(f"      Slave {j}: '{slave_name}' (type: {slave_type})")
                                    
                                    # Check for kidney-related names
                                    if any(keyword in slave_name.lower() for keyword in ['kidney', 'renal', 'right', 'left']):
                                        print(f"      *** POTENTIAL KIDNEY ANNOTATION ***")
                                        
                                        if hasattr(slave, 'data') and slave.data is not None:
                                            if hasattr(slave.data, 'shape'):
                                                print(f"        Mask shape: {slave.data.shape}")
                                            elif isinstance(slave.data, np.ndarray):
                                                print(f"        Mask shape: {slave.data.shape}")
                        else:
                            print(f"    Slaves: {type(img.slaves)} (not array)")
                    
                    print()  # Empty line for readability
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    training_dir = Path("../../../data/training")
    
    # Check files that might have annotations
    check_files = ["HemoB6M022_better.mat", "ExchangeB6M005.mat", "HemoB6M024.mat", "withoutROIwithMRI.mat"]
    
    for filename in check_files:
        filepath = training_dir / filename
        if filepath.exists():
            explore_images(filepath)

if __name__ == "__main__":
    main()
