"""
Quick utility to examine .mat file structure
"""
import scipy.io
import numpy as np

def examine_mat_file(filename):
    print(f"🔍 Examining: {filename}")
    print("="*50)
    
    try:
        data = scipy.io.loadmat(filename)
        
        print("📋 Keys in .mat file:")
        for key, value in data.items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, 'shape'):
                print(f"   {key}: {type(value).__name__} {value.shape}")
                if hasattr(value, 'dtype'):
                    print(f"      dtype: {value.dtype}")
                if len(value.shape) <= 3:
                    print(f"      range: [{np.min(value):.3f}, {np.max(value):.3f}]")
            else:
                print(f"   {key}: {type(value).__name__}")
                
        print("\n📊 Looking for MRI-like data (350x350x*):")
        for key, value in data.items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, 'shape') and len(value.shape) == 3:
                h, w, d = value.shape
                if h >= 300 and w >= 300:  # Relaxed criteria
                    print(f"   🎯 FOUND: {key} - {value.shape}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    examine_mat_file("withoutROIwithMRI.mat")
