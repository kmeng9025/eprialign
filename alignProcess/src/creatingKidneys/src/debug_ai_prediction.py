# #!/usr/bin/env python3
# """
# Debug AI prediction to understand why it's detecting the entire volume as kidney
# """

# import numpy as np
# import torch
# import scipy.io as sio
# from ai_kidney_detection import AIKidneyDetector

# def debug_ai_prediction():
#     """Debug the AI prediction step by step"""
    
#     print("🔍 DEBUGGING AI KIDNEY PREDICTION")
#     print("=" * 80)
    
#     # Load input file
#     input_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
    
#     print(f"📂 Loading input file: {input_file}")
    
#     # Initialize detector
#     detector = AIKidneyDetector()
    
#     # Load MRI data using detector's method
#     print(f"\n📊 Loading MRI data...")
    
#     try:
#         data = sio.loadmat(input_file)
#     except NotImplementedError:
#         # Handle MATLAB v7.3 format
#         import h5py
#         data = {}
#         with h5py.File(input_file, 'r') as f:
#             def traverse_datasets(name, obj):
#                 if isinstance(obj, h5py.Dataset):
#                     data[name] = obj[()]
#             f.visititems(traverse_datasets)
    
#     # Find MRI data in the file
#     mri_data = None
#     for key, value in data.items():
#         if not key.startswith('__') and hasattr(value, 'shape') and len(value.shape) == 3:
#             print(f"   Found 3D array '{key}': {value.shape}")
#             mri_data = value
#             break
    
#     if mri_data is None:
#         print("❌ No 3D MRI data found!")
#         return
    
#     print(f"✅ Using MRI data: {mri_data.shape}")
#     print(f"   Data type: {mri_data.dtype}")
#     print(f"   Value range: {np.min(mri_data)} to {np.max(mri_data)}")
#     print(f"   Total voxels: {np.prod(mri_data.shape):,}")
    
#     # Run AI prediction step by step
#     print(f"\n🧠 Running AI prediction...")
    
#     # Convert to tensor and normalize (like the model expects)
#     mri_tensor = torch.FloatTensor(mri_data).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
#     print(f"   Input tensor shape: {mri_tensor.shape}")
#     print(f"   Input tensor range: {torch.min(mri_tensor)} to {torch.max(mri_tensor)}")
    
#     # Run model prediction
#     with torch.no_grad():
#         detector.model.eval()
#         raw_output = detector.model(mri_tensor)
#         print(f"   Raw model output shape: {raw_output.shape}")
#         print(f"   Raw model output range: {torch.min(raw_output)} to {torch.max(raw_output)}")
        
#         # Apply sigmoid to get probabilities
#         probabilities = torch.sigmoid(raw_output)
#         print(f"   Probabilities range: {torch.min(probabilities)} to {torch.max(probabilities)}")
        
#         # Remove batch and channel dimensions
#         probabilities = probabilities.squeeze().numpy()
#         print(f"   Final probabilities shape: {probabilities.shape}")
        
#     print(f"\n🎯 Analyzing prediction results...")
    
#     # Check different confidence thresholds
#     thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
#     for threshold in thresholds:
#         kidney_mask = probabilities > threshold
#         kidney_voxels = np.count_nonzero(kidney_mask)
#         coverage = kidney_voxels / np.prod(probabilities.shape) * 100
#         print(f"   Threshold {threshold}: {kidney_voxels:,} voxels ({coverage:.2f}% coverage)")
    
#     # Show confidence distribution
#     print(f"\n📊 Confidence distribution:")
#     print(f"   Mean confidence: {np.mean(probabilities):.4f}")
#     print(f"   Median confidence: {np.median(probabilities):.4f}")
#     print(f"   Std confidence: {np.std(probabilities):.4f}")
#     print(f"   Min confidence: {np.min(probabilities):.4f}")
#     print(f"   Max confidence: {np.max(probabilities):.4f}")
    
#     # Check for potential issues
#     print(f"\n⚠️  POTENTIAL ISSUES:")
    
#     if np.mean(probabilities) > 0.8:
#         print(f"   🚨 VERY HIGH average confidence ({np.mean(probabilities):.4f}) - model might be overfitting")
    
#     if np.std(probabilities) < 0.1:
#         print(f"   🚨 LOW confidence variation ({np.std(probabilities):.4f}) - model not discriminating well")
    
#     if np.min(probabilities) > 0.5:
#         print(f"   🚨 ALL voxels above 0.5 confidence - model predicting everything as kidney!")
    
#     # Save prediction for inspection
#     output_file = "debug_prediction.npy"
#     np.save(output_file, probabilities)
#     print(f"\n💾 Saved prediction to: {output_file}")
    
#     return probabilities

# def check_training_data_sanity():
#     """Check if training data might have caused this issue"""
    
#     print(f"\n🔍 CHECKING TRAINING DATA SANITY")
#     print("=" * 80)
    
#     # Check if training data exists
#     training_file = "kidney_training_data.pkl"
#     if not os.path.exists(training_file):
#         print(f"❌ Training data file not found: {training_file}")
#         return
    
#     import pickle
    
#     with open(training_file, 'rb') as f:
#         training_data = pickle.load(f)
    
#     print(f"📊 Training data loaded:")
    
#     # Check what keys are actually in the training data
#     print(f"   Available keys: {list(training_data.keys())}")
    
#     # Look for MRI and kidney data with different possible names
#     mri_key = None
#     kidney_key = None
    
#     for key in training_data.keys():
#         if 'mri' in key.lower() or 'data' in key.lower() or 'volume' in key.lower():
#             mri_key = key
#         elif 'kidney' in key.lower() or 'mask' in key.lower() or 'label' in key.lower():
#             kidney_key = key
    
#     if mri_key:
#         print(f"   MRI data key: '{mri_key}', shape: {training_data[mri_key].shape}")
#     if kidney_key:
#         print(f"   Kidney data key: '{kidney_key}', shape: {training_data[kidney_key].shape}")
    
#     if not kidney_key:
#         print(f"❌ No kidney mask data found in training file")
#         return
    
#     # Check kidney mask properties
#     kidney_masks = training_data[kidney_key]
#     print(f"\n🎯 Training kidney mask analysis:")
    
#     for i in range(len(kidney_masks)):
#         mask = kidney_masks[i]
#         kidney_voxels = np.count_nonzero(mask)
#         total_voxels = np.prod(mask.shape)
#         coverage = kidney_voxels / total_voxels * 100
        
#         print(f"   Sample {i}: {kidney_voxels:,} kidney voxels ({coverage:.2f}% coverage)")
        
#         if coverage > 50:
#             print(f"      🚨 WARNING: Training sample {i} has {coverage:.2f}% kidney coverage - too high!")
#         elif coverage < 0.1:
#             print(f"      ⚠️  Warning: Training sample {i} has {coverage:.2f}% kidney coverage - very low")
#         else:
#             print(f"      ✅ Normal kidney coverage")

# if __name__ == "__main__":
#     import os
#     probabilities = debug_ai_prediction()
#     check_training_data_sanity()
