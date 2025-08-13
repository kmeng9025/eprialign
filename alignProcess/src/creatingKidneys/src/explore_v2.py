"""
Revised exploration with correct array indexing
"""
import numpy as np
import scipy.io
import os
from pathlib import Path

def explore_images_v2(filepath):
    """Explore images in a .mat file with corrected indexing"""
    print(f"\n{'='*50}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    try:
        data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        
        if 'images' in data:
            images = data['images']
            print(f"Images array shape: {images.shape}")
            print(f"Images array type: {type(images)}")
            
            # Handle different array shapes
            if isinstance(images, np.ndarray):
                if len(images.shape) == 1:  # 1D array
                    num_images = images.shape[0]
                    image_list = images
                elif len(images.shape) == 2:  # 2D array
                    num_images = images.shape[1]
                    image_list = images[0, :]  # take first row
                else:
                    print(f"Unexpected images shape: {images.shape}")
                    return
                
                print(f"Number of images: {num_images}")
                
                for i in range(num_images):
                    img = image_list[i] if len(images.shape) == 1 else images[0, i]
                    
                    if img is None or (isinstance(img, np.ndarray) and img.size == 0):
                        print(f"  Image {i}: Empty")
                        continue
                    
                    print(f"  Image {i}:")
                    
                    # Check for attributes
                    attrs = ['type', 'name', 'filename']
                    for attr in attrs:
                        if hasattr(img, attr):
                            val = getattr(img, attr)
                            print(f"    {attr.capitalize()}: {val}")
                    
                    # Check for data
                    if hasattr(img, 'data') and img.data is not None:
                        print(f"    Has data: {type(img.data)}")
                        
                        if hasattr(img.data, 'shape'):
                            print(f"    Data shape: {img.data.shape}")
                            
                            # Check for 3D data (potential MRI)
                            if len(img.data.shape) == 3:
                                print(f"    *** POTENTIAL MRI DATA (3D: {img.data.shape}) ***")
                                
                                # Check data type and range
                                if hasattr(img.data, 'dtype'):
                                    print(f"    Data type: {img.data.dtype}")
                                if hasattr(img.data, 'min') and hasattr(img.data, 'max'):
                                    try:
                                        print(f"    Data range: {img.data.min()} to {img.data.max()}")
                                    except:
                                        pass
                        
                        # Check for structured data
                        if hasattr(img.data, 'dtype') and img.data.dtype.names:
                            print(f"    Structured data fields: {list(img.data.dtype.names)}")
                    
                    # Check for slaves
                    if hasattr(img, 'slaves') and img.slaves is not None:
                        try:
                            if isinstance(img.slaves, np.ndarray):
                                if img.slaves.size > 0:
                                    print(f"    Slaves array shape: {img.slaves.shape}")
                                    
                                    # Try to iterate through slaves
                                    slaves_to_check = []
                                    if len(img.slaves.shape) == 1:
                                        slaves_to_check = img.slaves
                                    elif len(img.slaves.shape) == 2:
                                        slaves_to_check = img.slaves[0, :] if img.slaves.shape[0] > 0 else []
                                    
                                    for j, slave in enumerate(slaves_to_check):
                                        if slave is None or (isinstance(slave, np.ndarray) and slave.size == 0):
                                            continue
                                        
                                        slave_name = getattr(slave, 'name', 'Unnamed')
                                        slave_type = getattr(slave, 'type', 'Unknown')
                                        
                                        print(f"      Slave {j}: '{slave_name}' (type: {slave_type})")
                                        
                                        # Check for kidney-related annotations
                                        kidney_keywords = ['kidney', 'renal', 'left', 'right', 'LK', 'RK']
                                        if any(keyword.lower() in slave_name.lower() for keyword in kidney_keywords):
                                            print(f"      *** POTENTIAL KIDNEY ANNOTATION ***")
                                            
                                            if hasattr(slave, 'data') and slave.data is not None:
                                                if hasattr(slave.data, 'shape'):
                                                    print(f"        Annotation shape: {slave.data.shape}")
                                                    
                                                    # Check if it's binary mask
                                                    try:
                                                        unique_vals = np.unique(slave.data)
                                                        print(f"        Unique values: {unique_vals}")
                                                        if len(unique_vals) <= 3:  # Binary or small number of labels
                                                            print(f"        *** LOOKS LIKE BINARY MASK ***")
                                                    except:
                                                        pass
                                else:
                                    print(f"    Slaves: empty array")
                            else:
                                print(f"    Slaves: {type(img.slaves)} (not array)")
                        except Exception as e:
                            print(f"    Error processing slaves: {e}")
                    
                    print()  # Separator
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    training_dir = Path("../../../data/training")
    
    # Check all files to find which ones have annotations
    for mat_file in training_dir.glob("*.mat"):
        explore_images_v2(mat_file)

if __name__ == "__main__":
    main()
