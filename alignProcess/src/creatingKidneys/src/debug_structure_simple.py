"""
Quick debug script to understand the .mat file structure
"""
import scipy.io as sio
import numpy as np

def examine_structure(filename):
    print(f"🔍 EXAMINING: {filename}")
    print("="*50)
    
    data = sio.loadmat(filename)
    print(f"Top-level keys: {list(data.keys())}")
    
    if 'images' in data:
        images = data['images']
        print(f"Images type: {type(images)}")
        print(f"Images shape: {images.shape}")
        print(f"Images dtype: {images.dtype}")
        
        if images.size > 0:
            image0 = images[0, 0]  # Access first image
            print(f"First image type: {type(image0)}")
            print(f"First image dtype: {image0.dtype}")
            
            if hasattr(image0, 'dtype') and image0.dtype.names:
                print(f"Image fields: {image0.dtype.names}")
                
                if 'data' in image0.dtype.names:
                    mri_data = image0['data'][0, 0]
                    print(f"MRI data shape: {mri_data.shape}")
                    print(f"MRI data type: {type(mri_data)}")
                
                if 'slaves' in image0.dtype.names:
                    slaves = image0['slaves']
                    print(f"Slaves: {slaves}")

if __name__ == "__main__":
    examine_structure(r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat")
