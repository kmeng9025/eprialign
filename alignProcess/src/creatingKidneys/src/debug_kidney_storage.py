# #!/usr/bin/env python3
# """
# Debug script to examine kidney mask storage and validate actual kidney data
# """

# import numpy as np
# import scipy.io as sio
# import h5py
# import os

# def examine_kidney_storage():
#     """Examine the stored kidney data to check for issues"""
    
#     # Path to the AI-processed file
#     output_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\ai_kidneys_20250814_105021\withoutROIwithMRI_WITH_AI_KIDNEYS.mat"
    
#     if not os.path.exists(output_file):
#         print(f"❌ Output file does not exist: {output_file}")
#         return
    
#     print(f"🔍 Examining kidney storage in: {output_file}")
#     print("=" * 80)
    
#     try:
#         # Try loading with scipy first, fallback to h5py for v7.3 files
#         try:
#             data = sio.loadmat(output_file)
#         except NotImplementedError:
#             print("   File is MATLAB v7.3 format, using h5py...")
#             import h5py
#             data = {}
#             with h5py.File(output_file, 'r') as f:
#                 def traverse_datasets(name, obj):
#                     if isinstance(obj, h5py.Dataset):
#                         data[name] = obj[()]
#                 f.visititems(traverse_datasets)
        
#         print("📊 File contents:")
#         for key in data.keys():
#             if not key.startswith('__'):
#                 print(f"   {key}: {type(data[key])}")
#                 if hasattr(data[key], 'shape'):
#                     print(f"      Shape: {data[key].shape}")
#                 if hasattr(data[key], 'dtype'):
#                     print(f"      Type: {data[key].dtype}")
        
#         print("\n🔍 Looking for kidney-related data...")
        
#         # Check for various possible kidney data fields
#         kidney_fields = ['kidneys', 'kidney_masks', 'slaves', 'kidney_slaves', 'mask', 'segmentation']
#         found_kidneys = False
        
#         for field in kidney_fields:
#             if field in data:
#                 print(f"✅ Found kidney field: {field}")
#                 kidney_data = data[field]
#                 print(f"   Shape: {kidney_data.shape}")
#                 print(f"   Type: {kidney_data.dtype}")
#                 print(f"   Min value: {np.min(kidney_data)}")
#                 print(f"   Max value: {np.max(kidney_data)}")
#                 print(f"   Unique values: {np.unique(kidney_data)}")
#                 print(f"   Non-zero voxels: {np.count_nonzero(kidney_data)}")
#                 found_kidneys = True
        
#         # Check for Arbuz-specific slave structures
#         if 'img' in data:
#             img_data = data['img']
#             print(f"\n📊 Arbuz img structure:")
#             print(f"   Type: {type(img_data)}")
#             if hasattr(img_data, 'dtype') and img_data.dtype == 'object':
#                 print(f"   Contains {len(img_data.flatten())} objects")
#                 for i, obj in enumerate(img_data.flatten()):
#                     print(f"   Object {i}: {type(obj)}")
#                     if hasattr(obj, 'dtype') and obj.dtype.names:
#                         print(f"      Fields: {obj.dtype.names}")
#                         for field_name in obj.dtype.names:
#                             if 'slave' in field_name.lower() or 'kidney' in field_name.lower():
#                                 field_data = obj[field_name]
#                                 print(f"      {field_name}: {type(field_data)}, shape: {getattr(field_data, 'shape', 'N/A')}")
#                                 if hasattr(field_data, 'flatten'):
#                                     flat = field_data.flatten()
#                                     if len(flat) > 0:
#                                         print(f"         Content preview: {flat[0] if len(flat) > 0 else 'Empty'}")
        
#         # Check for any array with kidney-like dimensions
#         print(f"\n🔍 Checking all arrays for potential kidney masks...")
#         mri_shape = None
        
#         for key, value in data.items():
#             if not key.startswith('__') and hasattr(value, 'shape'):
#                 if len(value.shape) == 3:  # 3D array like MRI
#                     print(f"   3D array {key}: {value.shape}")
#                     if mri_shape is None:
#                         mri_shape = value.shape
#                         print(f"      (Assuming this is MRI shape: {mri_shape})")
                    
#                     # Check if this could be a kidney mask
#                     unique_vals = np.unique(value)
#                     nonzero_count = np.count_nonzero(value)
#                     if len(unique_vals) <= 10 and nonzero_count > 0:  # Likely binary/label mask
#                         print(f"      🎯 Potential kidney mask!")
#                         print(f"         Unique values: {unique_vals}")
#                         print(f"         Non-zero voxels: {nonzero_count}")
#                         print(f"         Coverage: {nonzero_count / np.prod(value.shape) * 100:.2f}%")
        
#         if not found_kidneys:
#             print("❌ No obvious kidney data found in standard fields")
        
#     except Exception as e:
#         print(f"❌ Error examining file: {e}")
#         import traceback
#         traceback.print_exc()

# def check_ai_pipeline_output():
#     """Check the AI pipeline intermediate outputs"""
    
#     print("\n🤖 Checking AI pipeline process...")
#     print("=" * 80)
    
#     # Load input file directly
#     input_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
    
#     try:
#         # Load input MRI data
#         try:
#             input_data = sio.loadmat(input_file)
#         except NotImplementedError:
#             import h5py
#             input_data = {}
#             with h5py.File(input_file, 'r') as f:
#                 def traverse_datasets(name, obj):
#                     if isinstance(obj, h5py.Dataset):
#                         input_data[name] = obj[()]
#                 f.visititems(traverse_datasets)
        
#         print(f"📊 Input file contents:")
#         for key in input_data.keys():
#             if not key.startswith('__'):
#                 value = input_data[key]
#                 print(f"   {key}: {type(value)}")
#                 if hasattr(value, 'shape'):
#                     print(f"      Shape: {value.shape}")
#                     if len(value.shape) == 3:
#                         print(f"      3D MRI data found: {value.shape}")
#                         print(f"      Min: {np.min(value)}, Max: {np.max(value)}")
                        
#     except Exception as e:
#         print(f"❌ Error loading input file: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     examine_kidney_storage()
#     check_ai_pipeline_output()
