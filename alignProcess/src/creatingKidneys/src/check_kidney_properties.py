#!/usr/bin/env python3
"""
Quick debug to check specific kidney slave properties
"""

import numpy as np
import h5py

def check_kidney_properties():
    """Check the specific properties of the kidney slave"""
    
    output_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\ai_kidneys_20250814_105021\withoutROIwithMRI_WITH_AI_KIDNEYS.mat"
    
    print("🔍 CHECKING KIDNEY SLAVE PROPERTIES")
    print("=" * 60)
    
    with h5py.File(output_file, 'r') as f:
        kidney_obj = f['#refs#/c']
        
        # Check key properties that might affect visibility
        print("🎯 Key Properties:")
        
        # Name
        name_data = kidney_obj['Name'][()]
        if hasattr(name_data, 'dtype') and name_data.dtype.char == 'U':
            name = ''.join(name_data)
        else:
            name = ''.join(chr(c) for c in name_data.flatten() if c != 0)
        print(f"   Name: '{name}'")
        
        # Visible flag
        visible = kidney_obj['Visible'][()][0, 0]
        print(f"   Visible: {visible}")
        
        # Selected flag  
        selected = kidney_obj['Selected'][()][0, 0]
        print(f"   Selected: {selected}")
        
        # isLoaded flag
        isLoaded = kidney_obj['isLoaded'][()][0, 0]
        print(f"   isLoaded: {isLoaded}")
        
        # isStore flag
        isStore = kidney_obj['isStore'][()][0, 0]
        print(f"   isStore: {isStore}")
        
        # Color
        color = kidney_obj['Color'][()]
        print(f"   Color: {color.flatten()}")
        
        # ImageType
        image_type_data = kidney_obj['ImageType'][()]
        if hasattr(image_type_data, 'dtype') and image_type_data.dtype.char == 'U':
            image_type = ''.join(image_type_data)
        else:
            image_type = ''.join(chr(c) for c in image_type_data.flatten() if c != 0)
        print(f"   ImageType: '{image_type}'")
        
        # Box dimensions
        box = kidney_obj['box'][()]
        print(f"   Box: {box.flatten()}")
        
        # Data shape and properties
        data = kidney_obj['data'][()]
        print(f"   Data shape: {data.shape}")
        print(f"   Data type: {data.dtype}")
        print(f"   Data range: {np.min(data)} to {np.max(data)}")
        print(f"   Non-zero voxels: {np.count_nonzero(data)}")
        
        # Check if this matches expected format
        print(f"\n✅ ANALYSIS:")
        print(f"   - Kidney is named: '{name}'")
        print(f"   - Visibility: {'ON' if visible == 1 else 'OFF'}")
        print(f"   - Selection: {'SELECTED' if selected == 1 else 'NOT SELECTED'}")
        print(f"   - Data loaded: {'YES' if isLoaded == 1 else 'NO'}")
        print(f"   - Data stored: {'YES' if isStore == 1 else 'NO'}")
        print(f"   - Contains: {np.count_nonzero(data):,} kidney voxels")
        
        if visible != 1:
            print(f"   ⚠️  ISSUE: Kidney slave is marked as NOT VISIBLE!")
        if isLoaded != 1:
            print(f"   ⚠️  ISSUE: Kidney data is marked as NOT LOADED!")
        if np.count_nonzero(data) == 0:
            print(f"   ❌ ISSUE: Kidney data contains no voxels!")
        else:
            print(f"   ✅ Kidney data contains proper voxels")

if __name__ == "__main__":
    check_kidney_properties()
