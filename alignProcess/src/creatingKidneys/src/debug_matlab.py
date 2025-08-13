#!/usr/bin/env python3
"""
Debug script to examine the structure of the MATLAB file
"""

import sys
from pathlib import Path
import scipy.io
import numpy as np

def examine_matlab_structure():
    """Examine the structure of the MATLAB file"""
    
    # Define path
    script_dir = Path(__file__).parent
    input_file = script_dir / "../../../data/training/withoutROIwithMRI.mat"
    input_file = input_file.resolve()
    
    print(f"Examining MATLAB file: {input_file}")
    
    # Load the file
    data = scipy.io.loadmat(str(input_file))
    
    print(f"Keys in file: {list(data.keys())}")
    
    # Examine images
    if 'images' in data:
        images = data['images']
        print(f"Images type: {type(images)}")
        print(f"Images shape: {images.shape if hasattr(images, 'shape') else 'No shape'}")
        
        if isinstance(images, np.ndarray):
            images_flat = images.flatten()
            print(f"Flattened images length: {len(images_flat)}")
            
            for i, img_item in enumerate(images_flat[:3]):  # Look at first 3
                print(f"\nImage {i}:")
                print(f"  Type: {type(img_item)}")
                print(f"  Shape: {img_item.shape if hasattr(img_item, 'shape') else 'No shape'}")
                
                if hasattr(img_item, 'dtype'):
                    print(f"  Dtype: {img_item.dtype}")
                
                if isinstance(img_item, np.ndarray) and img_item.size > 0:
                    print(f"  First element type: {type(img_item.item(0) if img_item.size > 0 else 'empty')}")
                    
                    # Try to extract the actual structure
                    try:
                        actual_img = img_item.item(0) if img_item.size > 0 else None
                        if actual_img is not None:
                            print(f"  Actual structure type: {type(actual_img)}")
                            if hasattr(actual_img, '_fieldnames'):
                                print(f"  Field names: {actual_img._fieldnames}")
                                for field in actual_img._fieldnames[:5]:  # First 5 fields
                                    try:
                                        value = getattr(actual_img, field)
                                        print(f"    {field}: {type(value)} - {value if isinstance(value, (str, int, float)) else 'complex'}")
                                    except:
                                        print(f"    {field}: <error accessing>")
                    except Exception as e:
                        print(f"  Error extracting: {e}")

if __name__ == "__main__":
    examine_matlab_structure()
