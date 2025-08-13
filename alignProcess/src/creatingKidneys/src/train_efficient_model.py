"""
Memory-efficient kidney segmentation model training
"""
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

def create_simple_features(mri_data, kidney_mask=None, sample_rate=0.01):
    """Create simple features with subsampling"""
    h, w, d = mri_data.shape
    
    # Sample only a subset of voxels to reduce memory usage
    total_voxels = h * w * d
    n_samples = int(total_voxels * sample_rate)
    
    # Random sampling indices
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(total_voxels, n_samples, replace=False)
    
    # Convert 1D indices to 3D coordinates
    z_coords = indices // (h * w)
    y_coords = (indices % (h * w)) // w
    x_coords = indices % w
    
    features = []
    labels = []
    
    for i in range(n_samples):
        x, y, z = x_coords[i], y_coords[i], z_coords[i]
        
        # Basic features
        intensity = mri_data[y, x, z]
        
        # Normalized position features
        pos_x = x / w
        pos_y = y / h
        pos_z = z / d
        
        # Distance from center
        center_x, center_y, center_z = w/2, h/2, d/2
        dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        dist_center_norm = dist_center / np.sqrt(center_x**2 + center_y**2 + center_z**2)
        
        # Local neighborhood (if possible)
        local_mean = intensity
        local_std = 0
        
        if 1 <= x < w-1 and 1 <= y < h-1 and 1 <= z < d-1:
            neighborhood = mri_data[y-1:y+2, x-1:x+2, z-1:z+2]
            local_mean = np.mean(neighborhood)
            local_std = np.std(neighborhood)
        
        feature_vector = [
            intensity,
            pos_x, pos_y, pos_z,
            dist_center_norm,
            local_mean,
            local_std
        ]
        
        features.append(feature_vector)
        
        if kidney_mask is not None:
            labels.append(kidney_mask[y, x, z])
    
    return np.array(features), np.array(labels) if kidney_mask is not None else None

def train_efficient_kidney_model():
    """Train an efficient kidney segmentation model"""
    print("🧠 EFFICIENT KIDNEY SEGMENTATION MODEL TRAINING")
    print("="*60)
    
    # Load training samples
    if not os.path.exists("kidney_training_samples.pkl"):
        print("❌ Training samples not found! Run extract_true_kidney_masks.py first.")
        return None
    
    with open("kidney_training_samples.pkl", 'rb') as f:
        samples = pickle.load(f)
    
    print(f"📦 Loaded {len(samples)} training samples")
    
    # Prepare training data with subsampling
    print("🔄 Preparing training data (with memory-efficient sampling)...")
    
    all_features = []
    all_labels = []
    sample_info = []
    
    # Use higher sampling rate for better model
    sample_rate = 0.05  # 5% of voxels
    
    for i, sample in enumerate(samples):
        print(f"  Processing sample {i+1}/{len(samples)}: {sample['file']}")
        
        mri_data = sample['mri_data'].astype(np.float32)
        kidney_mask = sample['kidney_mask'].astype(np.uint8)
        
        # Normalize MRI data
        if mri_data.max() > mri_data.min():
            mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
        
        # Create features with subsampling
        features, labels = create_simple_features(mri_data, kidney_mask, sample_rate)
        
        all_features.append(features)
        all_labels.append(labels)
        
        sample_info.append({
            'sample_id': i,
            'file': sample['file'],
            'shape': sample['mri_shape'],
            'coverage': sample['mask_coverage']
        })
        
        positive_rate = np.mean(labels)
        print(f"    Features: {features.shape}, Positive rate: {positive_rate:.4f}")
    
    # Combine all data
    print("🔗 Combining training data...")
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    
    print(f"📊 Total training data:")
    print(f"   Features: {X.shape}")
    print(f"   Labels: {y.shape}")
    print(f"   Positive samples: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
    
    # Split data
    print("✂️ Splitting into train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    
    # Train a simpler but effective model
    print("🚀 Training Logistic Regression model...")
    
    model = LogisticRegression(
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        max_iter=1000,
        solver='liblinear'  # Good for binary classification
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("📈 Evaluating model...")
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"   Training accuracy: {train_score:.4f}")
    print(f"   Validation accuracy: {val_score:.4f}")
    
    # Detailed metrics
    y_val_pred = model.predict(X_val)
    
    # Calculate custom metrics
    try:
        # Calculate Dice score manually
        intersection = np.sum((y_val == 1) & (y_val_pred == 1))
        dice = (2.0 * intersection) / (np.sum(y_val) + np.sum(y_val_pred)) if (np.sum(y_val) + np.sum(y_val_pred)) > 0 else 0
        
        # Calculate Jaccard (IoU) score manually
        union = np.sum((y_val == 1) | (y_val_pred == 1))
        jaccard = intersection / union if union > 0 else 0
        
        print(f"   Dice Score: {dice:.4f}")
        print(f"   Jaccard Score: {jaccard:.4f}")
        
        # Additional metrics
        precision = precision_score(y_val, y_val_pred, average='binary', zero_division=0)
        recall = recall_score(y_val, y_val_pred, average='binary', zero_division=0)
        f1 = f1_score(y_val, y_val_pred, average='binary', zero_division=0)
        
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        
    except Exception as e:
        print(f"   Could not calculate detailed metrics: {e}")
        dice = None
        jaccard = None
    
    # Feature importance (for logistic regression, use coefficients)
    feature_names = [
        'intensity', 'pos_x', 'pos_y', 'pos_z', 
        'dist_center', 'local_mean', 'local_std'
    ]
    
    coeffs = model.coef_[0]
    print(f"\n🔍 Feature Coefficients:")
    for name, coeff in zip(feature_names, coeffs):
        print(f"   {name}: {coeff:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"kidney_model_efficient_{timestamp}.joblib"
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'sample_rate': sample_rate,
        'training_info': {
            'n_samples': len(samples),
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'dice_score': dice if 'dice' in locals() else None,
            'jaccard_score': jaccard if 'jaccard' in locals() else None,
            'precision': precision if 'precision' in locals() else None,
            'recall': recall if 'recall' in locals() else None,
            'f1_score': f1 if 'f1' in locals() else None,
        },
        'sample_info': sample_info
    }
    
    joblib.dump(model_data, model_filename)
    
    print(f"\n💾 Model saved: {model_filename}")
    print(f"🎉 Training completed successfully!")
    
    return model_filename

def predict_kidney_mask(mri_data, model_data):
    """Predict kidney mask for given MRI data"""
    model = model_data['model']
    sample_rate = model_data.get('sample_rate', 0.05)
    
    # Normalize input
    if mri_data.max() > mri_data.min():
        mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
    
    # For prediction, we need to predict all voxels, not just a sample
    # So we'll predict in chunks or use full resolution
    h, w, d = mri_data.shape
    
    # Create features for all voxels (this could be memory intensive for very large volumes)
    print(f"Creating features for prediction on {h}x{w}x{d} volume...")
    
    prediction_mask = np.zeros((h, w, d), dtype=np.uint8)
    
    # Process slice by slice to manage memory
    for z in range(d):
        slice_features = []
        coords = []
        
        for y in range(h):
            for x in range(w):
                # Create features for this voxel
                intensity = mri_data[y, x, z]
                pos_x = x / w
                pos_y = y / h
                pos_z = z / d
                
                center_x, center_y, center_z = w/2, h/2, d/2
                dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                dist_center_norm = dist_center / np.sqrt(center_x**2 + center_y**2 + center_z**2)
                
                local_mean = intensity
                local_std = 0
                
                if 1 <= x < w-1 and 1 <= y < h-1 and 1 <= z < d-1:
                    neighborhood = mri_data[y-1:y+2, x-1:x+2, z-1:z+2]
                    local_mean = np.mean(neighborhood)
                    local_std = np.std(neighborhood)
                
                feature_vector = [
                    intensity, pos_x, pos_y, pos_z,
                    dist_center_norm, local_mean, local_std
                ]
                
                slice_features.append(feature_vector)
                coords.append((y, x))
        
        # Predict for this slice
        slice_features = np.array(slice_features)
        slice_predictions = model.predict(slice_features)
        
        # Fill in the prediction mask
        for i, (y, x) in enumerate(coords):
            prediction_mask[y, x, z] = slice_predictions[i]
        
        if (z + 1) % 5 == 0:
            print(f"  Processed slice {z+1}/{d}")
    
    return prediction_mask

if __name__ == "__main__":
    # Train the model
    model_file = train_efficient_kidney_model()
    
    if model_file:
        print(f"\n✅ MODEL TRAINING COMPLETE!")
        print(f"📁 Model file: {model_file}")
        print(f"🔗 Ready to integrate with pipeline!")
    else:
        print("❌ Training failed!")
