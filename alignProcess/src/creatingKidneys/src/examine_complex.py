"""
Advanced .mat file structure examiner for withoutROIwithMRI.mat
"""
import scipy.io
import numpy as np

def examine_complex_structure(obj, path="", depth=0, max_depth=5):
    """Recursively examine complex MATLAB structures"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    
    if hasattr(obj, 'dtype') and obj.dtype.names:
        print(f"{indent}📁 Structured array with fields: {obj.dtype.names}")
        
        # Try to access each field
        for field_name in obj.dtype.names:
            try:
                if obj.size > 0:
                    field_data = obj[field_name][0, 0] if obj.ndim >= 2 else obj[field_name]
                    print(f"{indent}  🔹 {field_name}: {type(field_data)}")
                    
                    if hasattr(field_data, 'shape'):
                        print(f"{indent}    Shape: {field_data.shape}")
                        
                        # Check if this looks like MRI data
                        if len(field_data.shape) == 3:
                            h, w, d = field_data.shape
                            if h >= 300 and w >= 300:  # Could be MRI
                                print(f"{indent}    🎯 POTENTIAL MRI: {field_data.shape}")
                                print(f"{indent}    Range: [{np.min(field_data):.3f}, {np.max(field_data):.3f}]")
                        
                        # Recurse if it's a structure
                        if hasattr(field_data, 'dtype') and field_data.dtype.names:
                            examine_complex_structure(field_data, f"{path}.{field_name}", depth + 1)
                
            except Exception as e:
                print(f"{indent}  ❌ Error accessing {field_name}: {e}")
    
    elif hasattr(obj, '__len__') and hasattr(obj, 'flat'):
        print(f"{indent}📦 Array with {len(obj.flat)} items")
        for i, item in enumerate(obj.flat[:3]):  # Examine first 3 items
            print(f"{indent}  Item {i}:")
            examine_complex_structure(item, f"{path}[{i}]", depth + 1)
            if i >= 2:  # Limit to first 3 items
                break

def examine_withoutROI():
    """Examine withoutROIwithMRI.mat specifically"""
    print("🔍 EXAMINING withoutROIwithMRI.mat")
    print("="*60)
    
    try:
        data = scipy.io.loadmat("withoutROIwithMRI.mat")
        
        print("📋 Top-level keys:")
        for key in data.keys():
            if not key.startswith('_'):
                print(f"  {key}")
        
        print("\n🔎 Examining 'sequences':")
        if 'sequences' in data:
            sequences = data['sequences']
            print(f"Sequences type: {type(sequences)}")
            print(f"Sequences shape: {sequences.shape}")
            examine_complex_structure(sequences, "sequences")
        
        print("\n🔎 Examining 'images':")
        if 'images' in data:
            images = data['images']
            print(f"Images type: {type(images)}")
            print(f"Images shape: {images.shape}")
            examine_complex_structure(images, "images")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    examine_withoutROI()
