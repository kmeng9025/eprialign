#!/usr/bin/env python3
"""
Debug ArbuzGUI slave structure to understand kidney visibility issues
"""

import numpy as np
import h5py
import os

def debug_arbuz_slaves():
    """Debug the ArbuzGUI slave structure"""
    
    output_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\ai_kidneys_20250814_105021\withoutROIwithMRI_WITH_AI_KIDNEYS.mat"
    
    print("🔍 DEBUGGING ARBUZGUI SLAVE STRUCTURE")
    print("=" * 80)
    
    with h5py.File(output_file, 'r') as f:
        print("🎯 Looking for kidney slave data...")
        
        # Examine image structures and their slaves
        if 'images' in f:
            images_ref = f['images']
            print(f"📊 Images reference: {images_ref}")
            
            # Follow the reference to the actual images
            for i, img_ref in enumerate(images_ref[0]):
                try:
                    # Handle h5py references
                    if isinstance(img_ref, h5py.h5r.Reference):
                        img_obj = f[img_ref]
                        print(f"\n📂 Image {i}: Reference resolved")
                        print(f"   Image object: {img_obj}")
                        
                        # Check for slaves
                        if 'slaves' in img_obj:
                            slaves_ref = img_obj['slaves']
                            print(f"   📄 Slaves: {slaves_ref}")
                            
                            # Check if slaves is a reference or direct data
                            slaves_data = slaves_ref[()]
                            print(f"   📄 Slaves data type: {type(slaves_data)}")
                            print(f"   📄 Slaves data: {slaves_data}")
                            
                            # If slaves_data contains references
                            if hasattr(slaves_data, '__iter__'):
                                for j, slave_ref in enumerate(slaves_data.flatten()):
                                    try:
                                        if isinstance(slave_ref, h5py.h5r.Reference):
                                            slave_obj = f[slave_ref]
                                            print(f"      🔗 Slave {j}: Reference resolved")
                                            print(f"         Slave object: {slave_obj}")
                                            
                                            # Examine slave properties
                                            for key in slave_obj.keys():
                                                try:
                                                    value = slave_obj[key][()]
                                                    print(f"         {key}: {type(value)} {getattr(value, 'shape', 'N/A')}")
                                                    
                                                    # Special attention to data and mask
                                                    if key == 'data' and hasattr(value, 'shape') and len(value.shape) == 3:
                                                        unique_vals = np.unique(value)
                                                        nonzero_count = np.count_nonzero(value)
                                                        print(f"            🎯 3D Data: unique={unique_vals}, nonzero={nonzero_count}")
                                                    
                                                    elif 'mask' in key.lower() and hasattr(value, 'shape'):
                                                        unique_vals = np.unique(value)
                                                        nonzero_count = np.count_nonzero(value)
                                                        print(f"            🎯 Mask: unique={unique_vals}, nonzero={nonzero_count}")
                                                except Exception as e:
                                                    print(f"         {key}: Error reading - {e}")
                                        else:
                                            print(f"      🔗 Slave {j}: Non-reference data {type(slave_ref)}")
                                    except Exception as e:
                                        print(f"      🔗 Slave {j}: Error - {e}")
                        else:
                            print(f"   ❌ No slaves found for image {i}")
                    
                    else:
                        print(f"\n📂 Image {i}: Non-reference data {type(img_ref)}")
                        
                except Exception as e:
                    print(f"\n📂 Image {i}: Error processing - {e}")
        
        # Also check top-level refs structure for the kidney data we found
        print(f"\n🔍 Examining kidney data in #refs#/c:")
        if '#refs#/c' in f:
            kidney_obj = f['#refs#/c']
            print(f"   Kidney object: {kidney_obj}")
            
            for key in kidney_obj.keys():
                value = kidney_obj[key][()]
                print(f"   {key}: {type(value)} {getattr(value, 'shape', 'N/A')}")
                
                if key == 'data' and hasattr(value, 'shape'):
                    unique_vals = np.unique(value)
                    nonzero_count = np.count_nonzero(value)
                    print(f"      🎯 Kidney Data: unique={unique_vals}, nonzero={nonzero_count}")
                    print(f"      🎯 Coverage: {nonzero_count / np.prod(value.shape) * 100:.2f}%")

def examine_matlab_kidney_structure():
    """Examine how create_kidney_slaves_final.m should have set up the structure"""
    
    print(f"\n🔧 MATLAB KIDNEY SLAVE STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Read the MATLAB script to understand expected structure
    matlab_script = r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\create_kidney_slaves_final.m"
    
    if os.path.exists(matlab_script):
        print("📄 Reading MATLAB script to understand expected structure...")
        with open(matlab_script, 'r') as f:
            content = f.read()
            
        # Look for key lines about slave creation
        lines = content.split('\n')
        print("🔍 Key MATLAB operations:")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['slave', 'kidney', 'mask', 'data', 'img']):
                print(f"   Line {i+1}: {line}")
    else:
        print("❌ MATLAB script not found")

if __name__ == "__main__":
    debug_arbuz_slaves()
    examine_matlab_kidney_structure()
