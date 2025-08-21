#!/usr/bin/env python3
"""
Inspect .mat file structure to understand data organization
"""

import os
import numpy as np
import scipy.io as sio

def inspect_mat_structure(filepath, max_depth=3, current_depth=0, prefix=""):
    """Recursively inspect the structure of a .mat file"""
    if current_depth > max_depth:
        return
    
    try:
        data = sio.loadmat(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return
    
    print(f"\n{'='*60}")
    print(f"INSPECTING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # Print top-level keys
    keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"Top-level keys: {keys}")
    
    for key in keys:
        print(f"\n{prefix}📁 Key: '{key}'")
        value = data[key]
        print(f"{prefix}   Type: {type(value)}")
        print(f"{prefix}   Shape: {getattr(value, 'shape', 'N/A')}")
        print(f"{prefix}   Dtype: {getattr(value, 'dtype', 'N/A')}")
        
        # If it's a structured array, show field names
        if hasattr(value, 'dtype') and value.dtype.names:
            print(f"{prefix}   Fields: {value.dtype.names}")
            
            # Inspect some common fields
            for field in ['images', 'slaves', 'sequences', 'data', 'name']:
                if field in value.dtype.names:
                    try:
                        field_data = value[field][0, 0]
                        print(f"{prefix}     🔍 {field}: type={type(field_data)}, shape={getattr(field_data, 'shape', 'N/A')}")
                        
                        # If it's an array of structures, show a few
                        if hasattr(field_data, 'shape') and len(field_data.shape) > 0 and field_data.shape[0] > 0:
                            print(f"{prefix}       Contains {field_data.shape[0]} items")
                            
                            # Look at first item
                            if field_data.shape[0] > 0:
                                first_item = field_data[0, 0] if len(field_data.shape) > 1 else field_data[0]
                                if hasattr(first_item, 'dtype') and first_item.dtype.names:
                                    print(f"{prefix}       First item fields: {first_item.dtype.names}")
                                    
                                    # Look for common subfields
                                    for subfield in ['name', 'data', 'mask']:
                                        if subfield in first_item.dtype.names:
                                            try:
                                                subfield_data = first_item[subfield][0, 0]
                                                if isinstance(subfield_data, np.ndarray) and subfield_data.size > 0:
                                                    if subfield == 'name':
                                                        name_str = str(subfield_data[0]) if subfield_data.ndim > 0 else str(subfield_data)
                                                        print(f"{prefix}         {subfield}: '{name_str}'")
                                                    else:
                                                        print(f"{prefix}         {subfield}: shape={subfield_data.shape}, type={subfield_data.dtype}")
                                            except:
                                                print(f"{prefix}         {subfield}: (error accessing)")
                    except Exception as e:
                        print(f"{prefix}     ❌ Error accessing {field}: {e}")

def main():
    """Inspect a few .mat files to understand structure"""
    training_data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
    
    # Get first few .mat files
    mat_files = []
    if os.path.exists(training_data_dir):
        for file in os.listdir(training_data_dir):
            if file.endswith('.mat'):
                mat_files.append(os.path.join(training_data_dir, file))
    
    # Inspect first 2 files to understand structure
    for i, mat_file in enumerate(mat_files[:2]):
        inspect_mat_structure(mat_file)
        
        if i == 0:  # For first file, also show detailed content
            print(f"\n{'='*40}")
            print("DETAILED INSPECTION OF FIRST FILE")
            print(f"{'='*40}")
            
            try:
                data = sio.loadmat(mat_file)
                main_keys = [k for k in data.keys() if not k.startswith('__')]
                
                if main_keys:
                    main_data = data[main_keys[0]]
                    print(f"Main data type: {type(main_data)}")
                    
                    if hasattr(main_data, 'dtype') and main_data.dtype.names:
                        print(f"Main data fields: {main_data.dtype.names}")
                        
                        # Show all fields in detail
                        for field_name in main_data.dtype.names:
                            try:
                                field_value = main_data[field_name][0, 0]
                                print(f"\n🔍 Field '{field_name}':")
                                print(f"   Type: {type(field_value)}")
                                print(f"   Shape: {getattr(field_value, 'shape', 'N/A')}")
                                
                                if isinstance(field_value, (str, np.str_)):
                                    print(f"   Value: '{field_value}'")
                                elif hasattr(field_value, 'shape') and len(field_value.shape) > 0:
                                    print(f"   Contains {field_value.shape[0]} items")
                                
                            except Exception as e:
                                print(f"   Error: {e}")
            except Exception as e:
                print(f"Detailed inspection error: {e}")

if __name__ == "__main__":
    main()
