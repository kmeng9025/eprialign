"""
Direct test to extract MRI from withoutROIwithMRI.mat
"""
import scipy.io
import numpy as np

def test_mri_extraction():
    print("🧪 TESTING MRI EXTRACTION")
    print("="*40)
    
    try:
        data = scipy.io.loadmat("withoutROIwithMRI.mat")
        
        # Get images array
        images = data['images']
        print(f"Images shape: {images.shape}")
        print(f"Images dtype: {images.dtype}")
        
        # Try to access first image
        print("\n🔍 Accessing first image:")
        first_image = images[0, 0]
        print(f"First image fields: {first_image.dtype.names}")
        
        # Access data field
        if 'data' in first_image.dtype.names:
            img_data = first_image['data']
            print(f"Image data shape: {img_data.shape}")
            
            # If it's a (1,1) array, extract the actual data
            if img_data.shape == (1, 1):
                actual_data = img_data[0, 0]
                if hasattr(actual_data, 'shape'):
                    print(f"Actual data shape: {actual_data.shape}")
                    print(f"Actual data range: [{np.min(actual_data):.3f}, {np.max(actual_data):.3f}]")
                    
                    # Test if this looks like MRI
                    if len(actual_data.shape) == 3 and actual_data.shape[0] == 350:
                        print("🎯 This looks like MRI data!")
                        return actual_data
            else:
                print(f"Image data range: [{np.min(img_data):.3f}, {np.max(img_data):.3f}]")
                
                # Test if this looks like MRI
                if len(img_data.shape) == 3 and img_data.shape[0] == 350:
                    print("🎯 This looks like MRI data!")
                    return img_data
        
        # Try second image
        print("\n🔍 Accessing second image:")
        if images.shape[1] > 1:
            second_image = images[0, 1]
            print(f"Second image fields: {second_image.dtype.names}")
            
            if 'data' in second_image.dtype.names:
                img_data = second_image['data']
                print(f"Image data shape: {img_data.shape}")
                
                # If it's a (1,1) array, extract the actual data
                if img_data.shape == (1, 1):
                    actual_data = img_data[0, 0]
                    if hasattr(actual_data, 'shape'):
                        print(f"Actual data shape: {actual_data.shape}")
                        print(f"Actual data range: [{np.min(actual_data):.3f}, {np.max(actual_data):.3f}]")
                        
                        if len(actual_data.shape) == 3 and actual_data.shape[0] == 350:
                            print("🎯 This looks like MRI data!")
                            return actual_data
                else:
                    print(f"Image data range: [{np.min(img_data):.3f}, {np.max(img_data):.3f}]")
                    
                    if len(img_data.shape) == 3 and img_data.shape[0] == 350:
                        print("🎯 This looks like MRI data!")
                        return img_data
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    mri_data = test_mri_extraction()
    if mri_data is not None:
        print(f"\n✅ Successfully extracted MRI: {mri_data.shape}")
    else:
        print(f"\n❌ Failed to extract MRI data")
