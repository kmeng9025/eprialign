# #!/usr/bin/env python3
# """
# Debug the resize-back process to see why entire volume is being marked as kidney
# """

# import numpy as np
# import torch
# from scipy.ndimage import zoom
# from ai_kidney_detection import AIKidneyDetector
# import scipy.io as sio

# def debug_resize_process():
#     """Debug the resize and resize-back process"""
    
#     print("🔍 DEBUGGING RESIZE PROCESS")
#     print("=" * 60)
    
#     # Load the actual data
#     input_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
#     data = sio.loadmat(input_file, struct_as_record=False, squeeze_me=True)
#     images = data['images']
#     mri_data = images[0].data
    
#     print(f"📊 Original MRI data:")
#     print(f"   Shape: {mri_data.shape}")
#     print(f"   Type: {mri_data.dtype}")
#     print(f"   Range: {np.min(mri_data)} to {np.max(mri_data)}")
    
#     # Step 1: Normalize
#     mri_normalized = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
#     print(f"\n📏 Normalized data:")
#     print(f"   Range: {np.min(mri_normalized):.4f} to {np.max(mri_normalized):.4f}")
#     print(f"   Mean: {np.mean(mri_normalized):.4f}")
    
#     # Step 2: Resize to model target
#     target_size = (64, 64, 32)
#     zoom_factors = [t/s for t, s in zip(target_size, mri_data.shape)]
#     print(f"\n📐 Resize to target:")
#     print(f"   Target size: {target_size}")
#     print(f"   Zoom factors: {zoom_factors}")
    
#     mri_resized = zoom(mri_normalized, zoom_factors, order=1)
#     print(f"   Resized shape: {mri_resized.shape}")
#     print(f"   Resized range: {np.min(mri_resized):.4f} to {np.max(mri_resized):.4f}")
    
#     # Step 3: Run model prediction
#     print(f"\n🧠 Running model prediction...")
#     detector = AIKidneyDetector()
    
#     input_tensor = torch.FloatTensor(mri_resized).unsqueeze(0).unsqueeze(0)
#     with torch.no_grad():
#         output = detector.model(input_tensor)
#         prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
    
#     print(f"   Model output shape: {prediction.shape}")
#     print(f"   Model output range: {np.min(prediction):.4f} to {np.max(prediction):.4f}")
#     print(f"   Model output mean: {np.mean(prediction):.4f}")
#     print(f"   Model output std: {np.std(prediction):.4f}")
    
#     # Check model output at different thresholds
#     print(f"\n🎯 Model output thresholding (64x64x32):")
#     thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
#     for threshold in thresholds:
#         mask = prediction > threshold
#         voxels = np.count_nonzero(mask)
#         coverage = voxels / np.prod(mask.shape) * 100
#         print(f"   Threshold {threshold}: {voxels:,} voxels ({coverage:.2f}% coverage)")
    
#     # Step 4: Resize back
#     print(f"\n📐 Resize back to original:")
#     original_zoom_factors = [s/t for s, t in zip(mri_data.shape, target_size)]
#     print(f"   Original zoom factors: {original_zoom_factors}")
    
#     kidney_prediction = zoom(prediction, original_zoom_factors, order=1)
#     print(f"   Resized back shape: {kidney_prediction.shape}")
#     print(f"   Resized back range: {np.min(kidney_prediction):.4f} to {np.max(kidney_prediction):.4f}")
#     print(f"   Resized back mean: {np.mean(kidney_prediction):.4f}")
#     print(f"   Resized back std: {np.std(kidney_prediction):.4f}")
    
#     # Check resize-back at different thresholds
#     print(f"\n🎯 Resized back thresholding (350x350x18):")
#     for threshold in thresholds:
#         mask = kidney_prediction > threshold
#         voxels = np.count_nonzero(mask)
#         coverage = voxels / np.prod(mask.shape) * 100
#         print(f"   Threshold {threshold}: {voxels:,} voxels ({coverage:.2f}% coverage)")
    
#     # Step 5: Final thresholding used by AI
#     threshold = 0.3
#     kidney_mask = kidney_prediction > threshold
    
#     print(f"\n✅ FINAL RESULT (threshold {threshold}):")
#     print(f"   Final mask shape: {kidney_mask.shape}")
#     print(f"   Final kidney voxels: {np.count_nonzero(kidney_mask):,}")
#     print(f"   Final coverage: {np.count_nonzero(kidney_mask) / np.prod(kidney_mask.shape) * 100:.2f}%")
    
#     # Save intermediate results for inspection
#     np.save("debug_original_mri.npy", mri_data)
#     np.save("debug_model_prediction.npy", prediction)
#     np.save("debug_resized_back.npy", kidney_prediction)
#     np.save("debug_final_mask.npy", kidney_mask)
    
#     print(f"\n💾 Saved debug files for inspection")
    
#     # Check if the model learned to predict everything as kidney
#     if np.mean(prediction) > 0.8:
#         print(f"\n🚨 MODEL ISSUE: Model predicting very high confidence ({np.mean(prediction):.3f}) across entire volume!")
#         print(f"   This suggests the model has learned to predict everything as kidney")
#         print(f"   Possible causes:")
#         print(f"   - Training data had incorrect labels")
#         print(f"   - Training data had kidney masks that were too large")
#         print(f"   - Model overfitted to predict high confidence everywhere")
    
#     return prediction, kidney_prediction, kidney_mask

# if __name__ == "__main__":
#     model_pred, resized_pred, final_mask = debug_resize_process()
