"""
Simple exploration of training data files
"""
import numpy as np
import scipy.io
import os
from pathlib import Path

def simple_explore(filepath):
    """Simple exploration of .mat file"""
    print(f"\n{'='*50}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    try:
        # Try different loading methods
        data = scipy.io.loadmat(filepath)
        print(f"Keys in file: {list(data.keys())}")
        
        # Remove matlab metadata
        real_keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Data keys: {real_keys}")
        
        for key in real_keys:
            val = data[key]
            print(f"  {key}: {type(val)} - {getattr(val, 'shape', 'no shape')}")
            
            if hasattr(val, 'dtype') and val.dtype.names:
                print(f"    Structured array fields: {val.dtype.names}")
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Try with different options
        try:
            data = scipy.io.loadmat(filepath, struct_as_record=False)
            real_keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"With struct_as_record=False - Keys: {real_keys}")
        except Exception as e2:
            print(f"Also failed with struct_as_record=False: {e2}")

def main():
    training_dir = Path("../../../data/training")
    
    # Test with a few files first
    test_files = ["withoutROIwithMRI.mat", "HemoB6M022_better.mat", "ExchangeB6M005.mat"]
    
    for filename in test_files:
        filepath = training_dir / filename
        if filepath.exists():
            simple_explore(filepath)
        else:
            print(f"File not found: {filename}")

if __name__ == "__main__":
    main()
