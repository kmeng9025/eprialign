# #!/usr/bin/env python3
# """
# Debug AI prediction with correct data loading
# """

# import numpy as np
# import torch
# import scipy.io as sio
# from ai_kidney_detection import AIKidneyDetector

# def debug_actual_ai_prediction():
#     """Debug using the same method as the AI detector"""
    
#     print("🔍 DEBUGGING AI PREDICTION - ACTUAL METHOD")
#     print("=" * 80)
    
#     input_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
    
#     print(f"📂 Loading with correct MATLAB parameters...")
    
#     # Use the same loading method as AI detector
#     data = sio.loadmat(input_file, struct_as_record=False, squeeze_me=True)
    
#     print(f"✅ Data loaded successfully")
#     print(f"Available fields: {list(data.keys())}")
    
#     if 'images' not in data:
#         print("❌ No 'images' field found!")
#         return
    
#     images = data['images']
#     print(f"📊 Images: {type(images)}, length: {len(images) if hasattr(images, '__len__') else 'N/A'}")
    
#     # Find MRI data (same as AI detector)
#     mri_data = None
#     for i in range(len(images)):
#         img = images[i]
#         print(f"\n📂 Image {i}: {type(img)}")
        
#         if hasattr(img, 'data') and img.data is not None:
#             print(f"   Has data: {img.data.shape if hasattr(img.data, 'shape') else 'No shape'}")
#             if hasattr(img.data, 'shape') and len(img.data.shape) == 3:
#                 mri_data = img.data
#                 print(f"   ✅ Found 3D MRI data: {mri_data.shape}")
#                 break
    
#     if mri_data is None:
#         print("❌ No 3D MRI data found!")
#         return
    
#     print(f"\n🎯 ANALYZING MRI DATA:")
#     print(f"   Shape: {mri_data.shape}")
#     print(f"   Type: {mri_data.dtype}")
#     print(f"   Range: {np.min(mri_data)} to {np.max(mri_data)}")
#     print(f"   Total voxels: {np.prod(mri_data.shape):,}")
    
#     # Now run the actual AI prediction
#     detector = AIKidneyDetector()
    
#     print(f"\n🧠 RUNNING AI PREDICTION...")
#     kidney_mask, num_kidneys, confidence = detector.predict_kidneys(mri_data)
    
#     print(f"📊 PREDICTION RESULTS:")
#     print(f"   Detected kidneys: {num_kidneys}")
#     print(f"   Confidence: {confidence:.4f}")
#     print(f"   Mask shape: {kidney_mask.shape}")
#     print(f"   Kidney voxels: {np.count_nonzero(kidney_mask):,}")
#     print(f"   Coverage: {np.count_nonzero(kidney_mask) / np.prod(kidney_mask.shape) * 100:.2f}%")
    
#     # Check the raw U-Net output
#     print(f"\n🔍 DETAILED U-NET ANALYSIS:")
    
#     # Normalize like the model does
#     mri_normalized = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
#     print(f"   Normalized range: {np.min(mri_normalized):.4f} to {np.max(mri_normalized):.4f}")
    
#     # Convert to tensor
#     input_tensor = torch.FloatTensor(mri_normalized).unsqueeze(0).unsqueeze(0)
#     print(f"   Input tensor shape: {input_tensor.shape}")
    
#     # Run raw model
#     with torch.no_grad():
#         detector.model.eval()
#         raw_output = detector.model(input_tensor)
#         probabilities = torch.sigmoid(raw_output).squeeze().numpy()
    
#     print(f"   Raw probabilities shape: {probabilities.shape}")
#     print(f"   Probabilities range: {np.min(probabilities):.4f} to {np.max(probabilities):.4f}")
#     print(f"   Mean probability: {np.mean(probabilities):.4f}")
#     print(f"   Std probability: {np.std(probabilities):.4f}")
    
#     # Test different thresholds
#     print(f"\n🎯 THRESHOLD ANALYSIS:")
#     thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
#     for threshold in thresholds:
#         mask = probabilities > threshold
#         voxels = np.count_nonzero(mask)
#         coverage = voxels / np.prod(mask.shape) * 100
#         print(f"   Threshold {threshold}: {voxels:,} voxels ({coverage:.2f}% coverage)")
    
#     # Save raw predictions for inspection
#     np.save("debug_raw_probabilities.npy", probabilities)
#     np.save("debug_mri_data.npy", mri_data)
#     print(f"\n💾 Saved raw data for inspection")
    
#     return probabilities, mri_data

# if __name__ == "__main__":
#     probabilities, mri_data = debug_actual_ai_prediction()
