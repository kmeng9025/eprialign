import scipy.io as sio
import numpy as np

def debug_mat_structure(file_path):
    print(f"🔍 Debugging: {file_path}")
    
    try:
        data = sio.loadmat(file_path)
        print(f"📋 Top-level keys: {list(data.keys())}")
        
        for key, value in data.items():
            if not key.startswith('__'):
                print(f"   {key}: {type(value)} - {getattr(value, 'shape', 'no shape')}")
                
                if isinstance(value, np.ndarray):
                    print(f"      dtype: {value.dtype}, ndim: {value.ndim}")
                    if value.ndim >= 3:
                        print(f"      ⭐ POTENTIAL MRI DATA: {value.shape}")
                        if min(value.shape) > 1:
                            print(f"         ✅ SUITABLE 3D DATA FOUND!")
        
        # Special handling for images array
        if 'images' in data:
            print(f"\n🖼️  Exploring images array...")
            images = data['images']
            print(f"   images shape: {images.shape}")
            
            for i in range(images.shape[1]):  # Loop through columns
                try:
                    img = images[0, i]
                    print(f"   Image {i}: {type(img)}")
                    
                    if hasattr(img, 'dtype'):
                        if img.dtype == 'object':
                            # It's a MATLAB struct
                            if hasattr(img, 'item'):
                                img_struct = img.item()
                                if hasattr(img_struct, 'dtype') and img_struct.dtype.names:
                                    fields = img_struct.dtype.names
                                    print(f"      Fields: {fields}")
                                    
                                    if 'data' in fields:
                                        img_data = img_struct['data'].item()
                                        if isinstance(img_data, np.ndarray):
                                            print(f"      data: {img_data.shape} {img_data.dtype}")
                                            if img_data.ndim >= 3 and min(img_data.shape) > 1:
                                                print(f"         ⭐ FOUND 3D MRI DATA: {img_data.shape}")
                                                
                except Exception as e:
                    print(f"      Error exploring image {i}: {e}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

# Test with training file
file_path = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
debug_mat_structure(file_path)
