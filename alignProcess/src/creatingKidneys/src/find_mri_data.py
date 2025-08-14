#!/usr/bin/env python3
"""
Simplified debug to find the actual MRI data in the input file
"""

import numpy as np
import scipy.io as sio
import h5py

def find_mri_data():
    """Find the actual MRI data in the input file"""
    
    input_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
    
    print("🔍 SEARCHING FOR MRI DATA")
    print("=" * 60)
    
    try:
        # Try scipy first
        data = sio.loadmat(input_file)
        print("📂 Loaded with scipy.io")
    except NotImplementedError:
        # Use h5py for v7.3 files
        print("📂 Loading with h5py (MATLAB v7.3)")
        data = {}
        with h5py.File(input_file, 'r') as f:
            def traverse_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
            f.visititems(traverse_datasets)
    
    print(f"\n🔍 Available data fields:")
    mri_candidates = []
    
    for key, value in data.items():
        if not key.startswith('__'):
            print(f"   {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"      Shape: {value.shape}")
                if len(value.shape) == 3:
                    mri_candidates.append((key, value))
                    print(f"      🎯 3D data candidate!")
            elif hasattr(value, '__len__'):
                print(f"      Length: {len(value)}")
    
    print(f"\n🎯 Found {len(mri_candidates)} 3D data candidates:")
    
    for key, value in mri_candidates:
        print(f"\n📊 Analyzing '{key}':")
        print(f"   Shape: {value.shape}")
        print(f"   Type: {value.dtype}")
        print(f"   Min: {np.min(value)}, Max: {np.max(value)}")
        print(f"   Total voxels: {np.prod(value.shape):,}")
        
        # Check if it looks like MRI data
        unique_vals = len(np.unique(value))
        print(f"   Unique values: {unique_vals}")
        
        if unique_vals > 100:  # Likely continuous MRI data
            print(f"   ✅ Looks like MRI data (many unique values)")
            return key, value
        else:
            print(f"   ⚠️  Might be mask data (few unique values)")
    
    return None, None

def check_ai_detector_data_loading():
    """Check how the AI detector loads data"""
    
    print(f"\n🤖 CHECKING AI DETECTOR DATA LOADING")
    print("=" * 60)
    
    from ai_kidney_detection import AIKidneyDetector
    
    detector = AIKidneyDetector()
    input_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
    
    # Look at the detector's _load_mri_data method
    print(f"📂 Using detector's method to load data...")
    
    # Try to mimic what the detector does
    try:
        # Check if detector has a _load_mri_data method
        if hasattr(detector, '_load_mri_data'):
            mri_data = detector._load_mri_data(input_file)
            print(f"✅ Detector loaded data: {mri_data.shape}")
        else:
            print("❌ Detector doesn't have _load_mri_data method")
            
            # Try to find the method in the AI detector code
            import inspect
            methods = inspect.getmembers(detector, predicate=inspect.ismethod)
            print(f"Available methods: {[name for name, _ in methods]}")
            
    except Exception as e:
        print(f"❌ Error loading with detector: {e}")

if __name__ == "__main__":
    key, value = find_mri_data()
    if key:
        print(f"\n✅ Best MRI candidate: '{key}' with shape {value.shape}")
    else:
        print(f"\n❌ No suitable MRI data found")
    
    check_ai_detector_data_loading()
