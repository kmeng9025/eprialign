"""
Examine the structure of withoutROIwithMRI.mat file
"""
import scipy.io as sio
import numpy as np

def examine_mat_file(mat_file):
    """Examine the structure of a .mat file"""
    print(f"🔍 EXAMINING: {mat_file}")
    print("="*60)
    
    try:
        mat_data = sio.loadmat(mat_file)
        print(f"📋 Found {len(mat_data)} keys in .mat file:")
        
        for key, value in mat_data.items():
            if not key.startswith('__'):
                if hasattr(value, 'shape'):
                    print(f"   {key:20s}: {value.shape} ({value.dtype})")
                    if len(value.shape) >= 3:
                        print(f"                         Min: {value.min():.3f}, Max: {value.max():.3f}")
                else:
                    print(f"   {key:20s}: {type(value)} - {str(value)[:50]}...")
        
        # Look for nested structures
        print(f"\n🔍 Looking for nested data...")
        for key, value in mat_data.items():
            if not key.startswith('__') and hasattr(value, 'dtype'):
                if value.dtype == 'object' or 'struct' in str(value.dtype):
                    print(f"\n📂 Examining nested structure: {key}")
                    if hasattr(value, 'shape') and value.shape == (1, 1):
                        nested = value[0, 0]
                        if hasattr(nested, 'dtype') and nested.dtype.names:
                            for field in nested.dtype.names:
                                field_data = nested[field]
                                if hasattr(field_data, 'shape'):
                                    print(f"   {field:20s}: {field_data.shape} ({field_data.dtype})")
                                    if len(field_data.shape) >= 3:
                                        print(f"                         Min: {field_data.min():.3f}, Max: {field_data.max():.3f}")
                                else:
                                    print(f"   {field:20s}: {type(field_data)}")
    
    except Exception as e:
        print(f"❌ Error examining file: {e}")

if __name__ == "__main__":
    mat_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
    examine_mat_file(mat_file)
