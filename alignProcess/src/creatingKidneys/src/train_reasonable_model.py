"""
Simplified kidney segmentation model training with reasonable sample sizes
"""
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

def extract_balanced_samples(mri_data, kidney_mask, n_samples_per_volume=5000):
    """Extract a balanced set of samples from each MRI volume"""
    h, w, d = mri_data.shape
    
    # Find kidney and non-kidney voxels
    kidney_voxels = np.where(kidney_mask == 1)
    non_kidney_voxels = np.where(kidney_mask == 0)
    
    # Get coordinates
    kidney_coords = list(zip(kidney_voxels[0], kidney_voxels[1], kidney_voxels[2]))
    non_kidney_coords = list(zip(non_kidney_voxels[0], non_kidney_voxels[1], non_kidney_voxels[2]))
    
    print(f"    Found {len(kidney_coords)} kidney voxels, {len(non_kidney_coords)} non-kidney voxels")
    
    # Sample balanced sets
    n_positive = min(len(kidney_coords), n_samples_per_volume // 2)
    n_negative = min(len(non_kidney_coords), n_samples_per_volume // 2)
    
    # Random sampling
    np.random.seed(42)
    if len(kidney_coords) > n_positive:
        kidney_sample_idx = np.random.choice(len(kidney_coords), n_positive, replace=False)
        kidney_coords = [kidney_coords[i] for i in kidney_sample_idx]
    
    if len(non_kidney_coords) > n_negative:
        non_kidney_sample_idx = np.random.choice(len(non_kidney_coords), n_negative, replace=False)
        non_kidney_coords = [non_kidney_coords[i] for i in non_kidney_sample_idx]
    
    # Combine coordinates and labels
    all_coords = kidney_coords + non_kidney_coords
    all_labels = [1] * len(kidney_coords) + [0] * len(non_kidney_coords)
    
    # Create features
    features = []
    for y, x, z in all_coords:
        # Simple but effective features
        intensity = mri_data[y, x, z]
        
        # Normalized position
        pos_x = x / w
        pos_y = y / h  
        pos_z = z / d
        
        # Distance from center
        center_x, center_y, center_z = w/2, h/2, d/2
        dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2 + center_z**2)
        dist_center_norm = dist_center / max_dist
        
        # Local statistics (3x3x3 neighborhood)
        local_mean = intensity
        local_std = 0
        
        if 1 <= x < w-1 and 1 <= y < h-1 and 1 <= z < d-1:
            neighborhood = mri_data[y-1:y+2, x-1:x+2, z-1:z+2]
            local_mean = np.mean(neighborhood)
            local_std = np.std(neighborhood)
        
        features.append([
            intensity,
            pos_x, pos_y, pos_z,
            dist_center_norm,
            local_mean,
            local_std
        ])
    
    return np.array(features), np.array(all_labels)

def train_kidney_model():
    """Train kidney segmentation model with reasonable sample size"""
    print("🧠 KIDNEY SEGMENTATION MODEL TRAINING")
    print("="*60)
    
    # Load training samples
    if not os.path.exists("kidney_training_samples.pkl"):
        print("❌ Training samples not found! Run extract_true_kidney_masks.py first.")
        return None
    
    with open("kidney_training_samples.pkl", 'rb') as f:
        samples = pickle.load(f)
    
    print(f"📦 Loaded {len(samples)} training samples")
    
    # Extract balanced samples from each volume
    print("🔄 Extracting balanced samples from each MRI volume...")
    
    all_features = []
    all_labels = []
    sample_info = []
    
    samples_per_volume = 10000  # Reasonable number per volume
    
    for i, sample in enumerate(samples):
        print(f"  Processing sample {i+1}/{len(samples)}: {sample['file']}")
        
        mri_data = sample['mri_data'].astype(np.float32)
        kidney_mask = sample['kidney_mask'].astype(np.uint8)
        
        # Normalize MRI data
        if mri_data.max() > mri_data.min():
            mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
        
        # Extract balanced samples
        features, labels = extract_balanced_samples(mri_data, kidney_mask, samples_per_volume)
        
        all_features.append(features)
        all_labels.append(labels)
        
        sample_info.append({
            'sample_id': i,
            'file': sample['file'],
            'shape': sample['mri_shape'],
            'coverage': sample['mask_coverage'],
            'n_samples': len(features)
        })
        
        positive_rate = np.mean(labels)
        print(f"    Extracted {len(features)} samples, positive rate: {positive_rate:.3f}")
    
    # Combine all data
    print("🔗 Combining training data...")
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    
    print(f"📊 Total training data:")
    print(f"   Features: {X.shape}")
    print(f"   Labels: {y.shape}")
    print(f"   Positive samples: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
    
    # This should now be manageable!
    if X.shape[0] > 200000:  # Still too many
        print("⚠️  Still too many samples, subsampling...")
        indices = np.random.choice(X.shape[0], 200000, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"   Reduced to: {X.shape}")
    
    # Split data
    print("✂️ Splitting into train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    
    # Train model
    print("🚀 Training Logistic Regression model...")
    
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("📈 Evaluating model...")
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"   Training accuracy: {train_score:.4f}")
    print(f"   Validation accuracy: {val_score:.4f}")
    
    # Detailed metrics
    y_val_pred = model.predict(X_val)
    
    precision = precision_score(y_val, y_val_pred, average='binary', zero_division=0)
    recall = recall_score(y_val, y_val_pred, average='binary', zero_division=0)
    f1 = f1_score(y_val, y_val_pred, average='binary', zero_division=0)
    
    # Dice score
    intersection = np.sum((y_val == 1) & (y_val_pred == 1))
    dice = (2.0 * intersection) / (np.sum(y_val) + np.sum(y_val_pred)) if (np.sum(y_val) + np.sum(y_val_pred)) > 0 else 0
    
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Dice Score: {dice:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"kidney_model_{timestamp}.joblib"
    
    model_data = {
        'model': model,
        'feature_names': ['intensity', 'pos_x', 'pos_y', 'pos_z', 'dist_center', 'local_mean', 'local_std'],
        'training_info': {
            'n_volumes': len(samples),
            'n_total_samples': X.shape[0],
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'dice_score': dice,
        },
        'sample_info': sample_info
    }
    
    joblib.dump(model_data, model_filename)
    
    print(f"\n💾 Model saved: {model_filename}")
    print(f"🎉 Training completed successfully!")
    
    return model_filename

if __name__ == "__main__":
    model_file = train_kidney_model()
    
    if model_file:
        print(f"\n✅ SUCCESS!")
        print(f"📁 Model: {model_file}")
        print(f"📊 Training summary:")
        print(f"   • Used {len(pickle.load(open('kidney_training_samples.pkl', 'rb')))} MRI volumes")
        print(f"   • Extracted balanced samples from each volume")
        print(f"   • Ready to integrate with pipeline!")
    else:
        print("❌ Training failed!")
