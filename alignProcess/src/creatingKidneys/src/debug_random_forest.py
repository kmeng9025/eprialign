"""
Debug Random Forest kidney detection to see what's happening
"""
import numpy as np
import scipy.io as sio
from random_forest_kidney import RandomForestKidneyModel
import os

def debug_random_forest_detection(mat_file):
    """Debug the Random Forest detection process"""
    print(f"🔍 DEBUGGING Random Forest Detection on: {mat_file}")
    print("="*60)
    
    # Load data
    print("📂 Loading MRI data...")
    mat_data = sio.loadmat(mat_file)
    
    # Find MRI data in the loaded structure
    mri_data = None
    print("   Looking for images array...")
    
    if 'images' in mat_data:
        images = mat_data['images']
        print(f"   Found images array: {images.shape}")
        
        # Try to access the first image
        if images.size > 0:
            first_image = images[0, 0]
            print(f"   First image type: {type(first_image)}")
            
            # Look for MRI data in the image structure
            if hasattr(first_image, 'dtype') and first_image.dtype.names:
                print("   Image fields:", first_image.dtype.names)
                for field in first_image.dtype.names:
                    field_data = first_image[field]
                    if hasattr(field_data, 'shape'):
                        print(f"     {field}: {field_data.shape}")
                        # Look for 3D data that could be MRI
                        if len(field_data.shape) == 3 and field_data.shape[0] > 100:
                            mri_data = field_data[0, 0] if field_data.shape == (1, 1) else field_data
                            print(f"   ✅ Using MRI data from {field}: {mri_data.shape}")
                            break
                        elif len(field_data.shape) == 2 and field_data.shape == (1, 1):
                            # Nested structure
                            nested_data = field_data[0, 0]
                            if hasattr(nested_data, 'shape') and len(nested_data.shape) == 3:
                                mri_data = nested_data
                                print(f"   ✅ Using nested MRI data from {field}: {mri_data.shape}")
                                break
    
    if mri_data is None:
        print("   Trying to find any 3D data...")
        for key in mat_data.keys():
            if not key.startswith('__'):
                data = mat_data[key]
                if hasattr(data, 'shape') and len(data.shape) >= 3:
                    print(f"   Found potential MRI data: {key} with shape {data.shape}")
                    if data.shape[0] > 100 and data.shape[1] > 100:  # Likely MRI dimensions
                        mri_data = data
                        print(f"   ✅ Using MRI data: {key} with shape {data.shape}")
                        break
    
    if mri_data is None:
        print("❌ Could not find MRI data!")
        return
    
    # Load Random Forest model
    print("\n🌳 Loading Random Forest model...")
    model_path = "kidney_random_forest_best.joblib"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    model = RandomForestKidneyModel(model_path)
    print("   ✅ Model loaded successfully")
    
    # Generate candidate regions
    print(f"\n🔬 Analyzing MRI volume: {mri_data.shape}")
    print(f"   Intensity range: {mri_data.min():.3f} - {mri_data.max():.3f}")
    
    candidates = model.generate_candidate_regions(mri_data)
    print(f"   📊 Generated {len(candidates)} candidate regions")
    
    if len(candidates) == 0:
        print("   ❌ No candidate regions found!")
        print("   💡 Try adjusting thresholds in generate_candidate_regions()")
        return
    
    # Analyze candidates
    print(f"\n📋 Candidate Region Analysis:")
    for i, candidate in enumerate(candidates[:10]):  # Show first 10
        print(f"   {i+1:2d}. Size: {candidate['size']:6d} voxels, Method: {candidate['method']}")
    
    # Extract features and predict
    print(f"\n🧠 Running Random Forest prediction...")
    features_list = []
    for candidate in candidates:
        features = model.extract_region_features(mri_data, candidate['mask'])
        features_list.append(features)
    
    if len(features_list) == 0:
        print("   ❌ No features extracted!")
        return
    
    X = np.array(features_list)
    X_scaled = model.scaler.transform(X)
    
    predictions = model.model.predict(X_scaled)
    probabilities = model.model.predict_proba(X_scaled)[:, 1]
    
    # Analyze predictions
    print(f"\n📊 Prediction Results:")
    print(f"   Total regions: {len(predictions)}")
    print(f"   Predicted kidneys: {np.sum(predictions)}")
    print(f"   Max probability: {np.max(probabilities):.3f}")
    print(f"   Min probability: {np.min(probabilities):.3f}")
    print(f"   Mean probability: {np.mean(probabilities):.3f}")
    
    # Show top predictions
    sorted_indices = np.argsort(probabilities)[::-1]
    print(f"\n🏆 Top 5 Predictions:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        prob = probabilities[idx]
        pred = predictions[idx]
        size = candidates[idx]['size']
        method = candidates[idx]['method']
        print(f"   {i+1}. Prob: {prob:.3f}, Pred: {pred}, Size: {size:6d}, Method: {method}")
    
    # Apply thresholds
    print(f"\n🎯 Threshold Analysis:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        kidney_regions = np.sum((predictions == 1) & (probabilities > threshold))
        print(f"   Threshold {threshold:.1f}: {kidney_regions} kidney regions")

if __name__ == "__main__":
    mat_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
    debug_random_forest_detection(mat_file)
