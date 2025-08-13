#!/usr/bin/env python3
"""
Debug script to understand how to properly modify the MATLAB structure
"""

import sys
from pathlib import Path
import scipy.io
import numpy as np

def debug_project_structure():
    """Debug the project structure to understand modification approach"""
    
    # Define path
    script_dir = Path(__file__).parent
    input_file = script_dir / "../../../data/training/withoutROIwithMRI.mat"
    input_file = input_file.resolve()
    
    print(f"Debugging project structure: {input_file}")
    
    # Load the file
    data = scipy.io.loadmat(str(input_file))
    
    print(f"Top level keys: {list(data.keys())}")
    
    # Examine images structure
    images = data['images']
    print(f"\nImages:")
    print(f"  Type: {type(images)}")
    print(f"  Shape: {images.shape}")
    print(f"  Dtype: {images.dtype}")
    
    # Try to understand how to modify it
    print(f"\nTrying to access images[0,1]:")
    try:
        img_1 = images[0, 1]
        print(f"  Success: type={type(img_1)}, shape={img_1.shape}")
        
        # Extract the structure
        img_struct = img_1.item(0)
        field_names = img_1.dtype.names
        print(f"  Field names: {field_names}")
        
        # Look at slaves field specifically
        slaves_idx = list(field_names).index('slaves')
        slaves_data = img_struct[slaves_idx]
        print(f"  Slaves field: type={type(slaves_data)}, value={slaves_data}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test creating a modified copy
    print(f"\nTesting modification approach:")
    try:
        # Create a copy
        new_images = images.copy()
        print(f"  Copy created successfully: {new_images.shape}")
        
        # Try to modify the slaves field
        img_to_modify = new_images[0, 1]
        img_struct = img_to_modify.item(0)
        
        # Convert to list for modification
        img_list = list(img_struct)
        slaves_idx = list(img_to_modify.dtype.names).index('slaves')
        
        # Create dummy slave data
        dummy_slave = np.array([([],  # FileName
                                'LeftKidney',  # Name
                                [],  # isStore
                                [],  # isLoaded
                                [],  # Visible
                                [],  # Selected
                                '3DMASK',  # ImageType
                                np.zeros((10, 10)),  # data
                                [],  # data_info
                                [],  # box
                                np.eye(4),  # Anative
                                np.eye(4),  # A
                                [],  # Aprime
                                [],  # slaves
                                [],  # Apre
                                [])], dtype=img_to_modify.dtype)  # Anext
        
        print(f"  Created dummy slave structure")
        
    except Exception as e:
        print(f"  Modification test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_project_structure()
