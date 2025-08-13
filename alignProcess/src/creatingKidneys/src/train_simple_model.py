"""
Simple but effective kidney segmentation model training
"""
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

def create_features_from_mri(mri_data):
    """Create features from MRI data for traditional ML"""
    # Flatten the 3D volume to 2D for traditional ML
    features = []
    
    # Get shape
    h, w, d = mri_data.shape
    
    # Create features for each voxel
    for z in range(d):
        for y in range(h):
            for x in range(w):
                # Basic features
                intensity = mri_data[y, x, z]
                
                # Position features (normalized)
                pos_x = x / w
                pos_y = y / h
                pos_z = z / d
                
                # Local neighborhood features (if not on border)
                local_mean = intensity
                local_std = 0
                local_max = intensity
                local_min = intensity
                
                if 1 <= x < w-1 and 1 <= y < h-1 and 1 <= z < d-1:
                    neighborhood = mri_data[y-1:y+2, x-1:x+2, z-1:z+2]
                    local_mean = np.mean(neighborhood)
                    local_std = np.std(neighborhood)
                    local_max = np.max(neighborhood)
                    local_min = np.min(neighborhood)
                
                # Gradient features (simple differences)
                grad_x = 0
                grad_y = 0
                grad_z = 0
                
                if x > 0:
                    grad_x = mri_data[y, x, z] - mri_data[y, x-1, z]
                if y > 0:
                    grad_y = mri_data[y, x, z] - mri_data[y-1, x, z]
                if z > 0:
                    grad_z = mri_data[y, x, z] - mri_data[y, x, z-1]
                
                feature_vector = [
                    intensity,           # Original intensity
                    pos_x, pos_y, pos_z, # Position features
                    local_mean,          # Local statistics
                    local_std,
                    local_max,
                    local_min,
                    grad_x, grad_y, grad_z  # Gradient features
                ]
                
                features.append(feature_vector)
    
    return np.array(features)

def train_simple_kidney_model():
    """Train a simple but effective kidney segmentation model"""
    print("🧠 KIDNEY SEGMENTATION MODEL TRAINING")
    print("="*60)
    
    # Load training samples
    if not os.path.exists("kidney_training_samples.pkl"):
        print("❌ Training samples not found! Run extract_true_kidney_masks.py first.")
        return None
    
    with open("kidney_training_samples.pkl", 'rb') as f:
        samples = pickle.load(f)
    
    print(f"📦 Loaded {len(samples)} training samples")
    
    # Prepare training data
    print("🔄 Preparing training data...")
    
    all_features = []
    all_labels = []
    sample_info = []
    
    for i, sample in enumerate(samples):
        print(f"  Processing sample {i+1}/{len(samples)}: {sample['file']}")
        
        mri_data = sample['mri_data'].astype(np.float32)
        kidney_mask = sample['kidney_mask'].astype(np.uint8)
        
        # Normalize MRI data
        if mri_data.max() > mri_data.min():
            mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
        
        # Create features
        features = create_features_from_mri(mri_data)
        labels = kidney_mask.flatten()
        
        all_features.append(features)
        all_labels.append(labels)
        
        sample_info.append({
            'sample_id': i,
            'file': sample['file'],
            'shape': sample['mri_shape'],
            'coverage': sample['mask_coverage']
        })
        
        print(f"    Features shape: {features.shape}, Labels: {np.sum(labels)} positive voxels")
    
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
    
    # Train Random Forest model (good for medical segmentation)
    print("🌲 Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,          # Good balance of performance and speed
        max_depth=20,              # Prevent overfitting
        min_samples_split=10,      # Require minimum samples to split
        min_samples_leaf=5,        # Require minimum samples in leaf
        class_weight='balanced',   # Handle class imbalance
        random_state=42,
        n_jobs=-1,                 # Use all CPU cores
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("📈 Evaluating model...")
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"   Training accuracy: {train_score:.4f}")
    print(f"   Validation accuracy: {val_score:.4f}")
    
    # Predictions for more detailed metrics
    y_val_pred = model.predict(X_val)
    
    # Calculate custom metrics for medical segmentation
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
    
    # Feature importance
    feature_names = [
        'intensity', 'pos_x', 'pos_y', 'pos_z', 
        'local_mean', 'local_std', 'local_max', 'local_min',
        'grad_x', 'grad_y', 'grad_z'
    ]
    
    importance = model.feature_importances_
    print(f"\n🔍 Feature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"   {name}: {imp:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"kidney_model_{timestamp}.joblib"
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_info': {
            'n_samples': len(samples),
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'dice_score': dice if 'dice' in locals() else None,
            'jaccard_score': jaccard if 'jaccard' in locals() else None,
        },
        'sample_info': sample_info
    }
    
    joblib.dump(model_data, model_filename)
    
    print(f"\n💾 Model saved: {model_filename}")
    print(f"🎉 Training completed successfully!")
    
    return model_filename

def test_model_on_sample():
    """Test the trained model on a sample"""
    print("\n🧪 TESTING MODEL ON SAMPLE")
    print("="*40)
    
    # Load the most recent model
    model_files = sorted([f for f in os.listdir('.') if f.startswith('kidney_model_') and f.endswith('.joblib')])
    if not model_files:
        print("❌ No trained model found!")
        return
    
    latest_model = model_files[-1]
    print(f"📂 Loading model: {latest_model}")
    
    model_data = joblib.load(latest_model)
    model = model_data['model']
    
    # Load training samples for testing
    with open("kidney_training_samples.pkl", 'rb') as f:
        samples = pickle.load(f)
    
    # Test on first sample
    test_sample = samples[0]
    print(f"🎯 Testing on: {test_sample['file']}")
    
    mri_data = test_sample['mri_data'].astype(np.float32)
    true_mask = test_sample['kidney_mask']
    
    # Normalize
    if mri_data.max() > mri_data.min():
        mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
    
    # Create features
    features = create_features_from_mri(mri_data)
    
    # Predict
    print("🔮 Making predictions...")
    predictions = model.predict(features)
    pred_mask = predictions.reshape(true_mask.shape)
    
    # Calculate metrics
    true_positive = np.sum((pred_mask == 1) & (true_mask == 1))
    false_positive = np.sum((pred_mask == 1) & (true_mask == 0))
    false_negative = np.sum((pred_mask == 0) & (true_mask == 1))
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    print(f"📊 Test Results:")
    print(f"   True kidney voxels: {np.sum(true_mask)}")
    print(f"   Predicted kidney voxels: {np.sum(pred_mask)}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    
    return latest_model

if __name__ == "__main__":
    # Install required packages
    try:
        import sklearn
        import joblib
    except ImportError:
        print("Installing required packages...")
        os.system("pip install scikit-learn joblib")
        import sklearn
        import joblib
    
    # Train the model
    model_file = train_simple_kidney_model()
    
    if model_file:
        # Test the model
        test_model_on_sample()
        
        print(f"\n✅ MODEL TRAINING COMPLETE!")
        print(f"📁 Model file: {model_file}")
        print(f"🔗 Ready to integrate with pipeline!")
    else:
        print("❌ Training failed!")
