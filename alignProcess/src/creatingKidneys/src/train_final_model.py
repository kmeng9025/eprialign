"""
Simple kidney segmentation model training using extracted data
"""
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

def train_kidney_model():
    """Train kidney segmentation model on extracted data"""
    print("🧠 KIDNEY SEGMENTATION MODEL TRAINING")
    print("="*50)
    
    # Load training data
    print("📂 Loading training data...")
    try:
        with open("kidney_training_data.pkl", 'rb') as f:
            training_data = pickle.load(f)
        
        mri_volumes = training_data['mri_volumes']
        kidney_masks = training_data['kidney_masks']
        metadata = training_data['metadata']
        
        print(f"✅ Loaded {len(mri_volumes)} training volumes")
        
    except FileNotFoundError:
        print("❌ Training data not found! Run extract_training_arrays.py first.")
        return None
    
    # Create training samples by sampling from each volume
    print("🔄 Creating training samples...")
    
    all_features = []
    all_labels = []
    
    samples_per_volume = 5000  # Reasonable number per volume
    
    for i, (mri, mask, meta) in enumerate(zip(mri_volumes, kidney_masks, metadata)):
        print(f"  Processing volume {i+1}/{len(mri_volumes)}: {meta['file']}")
        
        h, w, d = mri.shape
        
        # Find kidney and non-kidney voxels
        kidney_voxels = np.where(mask == 1)
        non_kidney_voxels = np.where(mask == 0)
        
        n_kidney = len(kidney_voxels[0])
        n_non_kidney = len(non_kidney_voxels[0])
        
        print(f"    {n_kidney} kidney voxels, {n_non_kidney} non-kidney voxels")
        
        # Sample balanced sets
        n_pos_samples = min(n_kidney, samples_per_volume // 2)
        n_neg_samples = min(n_non_kidney, samples_per_volume // 2)
        
        # Random sampling
        np.random.seed(42 + i)  # Different seed per volume
        
        # Sample kidney voxels
        if n_kidney > n_pos_samples:
            kidney_indices = np.random.choice(n_kidney, n_pos_samples, replace=False)
        else:
            kidney_indices = np.arange(n_kidney)
        
        # Sample non-kidney voxels
        if n_non_kidney > n_neg_samples:
            non_kidney_indices = np.random.choice(n_non_kidney, n_neg_samples, replace=False)
        else:
            non_kidney_indices = np.arange(n_non_kidney)
        
        # Create feature vectors
        volume_features = []
        volume_labels = []
        
        # Process kidney samples
        for idx in kidney_indices:
            y, x, z = kidney_voxels[0][idx], kidney_voxels[1][idx], kidney_voxels[2][idx]
            
            # Simple features
            intensity = mri[y, x, z]
            pos_x = x / w
            pos_y = y / h
            pos_z = z / d
            
            # Distance from center
            center_x, center_y, center_z = w/2, h/2, d/2
            dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2 + center_z**2)
            dist_center_norm = dist_center / max_dist
            
            # Local mean (3x3x3 neighborhood)
            local_mean = intensity
            if 1 <= x < w-1 and 1 <= y < h-1 and 1 <= z < d-1:
                local_mean = np.mean(mri[y-1:y+2, x-1:x+2, z-1:z+2])
            
            features = [intensity, pos_x, pos_y, pos_z, dist_center_norm, local_mean]
            volume_features.append(features)
            volume_labels.append(1)
        
        # Process non-kidney samples
        for idx in non_kidney_indices:
            y, x, z = non_kidney_voxels[0][idx], non_kidney_voxels[1][idx], non_kidney_voxels[2][idx]
            
            intensity = mri[y, x, z]
            pos_x = x / w
            pos_y = y / h
            pos_z = z / d
            
            center_x, center_y, center_z = w/2, h/2, d/2
            dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2 + center_z**2)
            dist_center_norm = dist_center / max_dist
            
            local_mean = intensity
            if 1 <= x < w-1 and 1 <= y < h-1 and 1 <= z < d-1:
                local_mean = np.mean(mri[y-1:y+2, x-1:x+2, z-1:z+2])
            
            features = [intensity, pos_x, pos_y, pos_z, dist_center_norm, local_mean]
            volume_features.append(features)
            volume_labels.append(0)
        
        all_features.extend(volume_features)
        all_labels.extend(volume_labels)
        
        print(f"    Created {len(volume_features)} samples ({np.sum(volume_labels)} positive)")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n📊 Training data summary:")
    print(f"   Total samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    
    # Split into train/validation
    print("✂️ Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    
    # Train model
    print("🚀 Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("📈 Evaluating model...")
    
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    y_val_pred = model.predict(X_val)
    
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    
    # Dice score
    intersection = np.sum((y_val == 1) & (y_val_pred == 1))
    dice = (2.0 * intersection) / (np.sum(y_val) + np.sum(y_val_pred)) if (np.sum(y_val) + np.sum(y_val_pred)) > 0 else 0
    
    print(f"   Training accuracy: {train_acc:.4f}")
    print(f"   Validation accuracy: {val_acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Dice Score: {dice:.4f}")
    
    # Feature importance
    feature_names = ['intensity', 'pos_x', 'pos_y', 'pos_z', 'dist_center', 'local_mean']
    importance = model.feature_importances_
    
    print(f"\n🔍 Feature importance:")
    for name, imp in zip(feature_names, importance):
        print(f"   {name}: {imp:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"kidney_model_{timestamp}.joblib"
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_info': {
            'n_volumes': len(mri_volumes),
            'n_samples': X.shape[0],
            'train_acc': train_acc,
            'val_acc': val_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'dice_score': dice,
        }
    }
    
    joblib.dump(model_data, model_filename)
    
    print(f"\n💾 Model saved: {model_filename}")
    print(f"🎉 Training completed!")
    
    return model_filename

def predict_kidney_mask(mri_volume, model_data):
    """Predict kidney mask for a given MRI volume"""
    model = model_data['model']
    
    h, w, d = mri_volume.shape
    prediction = np.zeros((h, w, d), dtype=np.uint8)
    
    print(f"🔮 Predicting kidney mask for {h}×{w}×{d} volume...")
    
    # Process slice by slice to manage memory
    for z in range(d):
        slice_features = []
        coords = []
        
        for y in range(h):
            for x in range(w):
                intensity = mri_volume[y, x, z]
                pos_x = x / w
                pos_y = y / h
                pos_z = z / d
                
                center_x, center_y, center_z = w/2, h/2, d/2
                dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2 + center_z**2)
                dist_center_norm = dist_center / max_dist
                
                local_mean = intensity
                if 1 <= x < w-1 and 1 <= y < h-1 and 1 <= z < d-1:
                    local_mean = np.mean(mri_volume[y-1:y+2, x-1:x+2, z-1:z+2])
                
                features = [intensity, pos_x, pos_y, pos_z, dist_center_norm, local_mean]
                slice_features.append(features)
                coords.append((y, x))
        
        # Predict slice
        slice_features = np.array(slice_features)
        slice_predictions = model.predict(slice_features)
        
        # Fill prediction mask
        for i, (y, x) in enumerate(coords):
            prediction[y, x, z] = slice_predictions[i]
        
        if (z + 1) % 10 == 0:
            print(f"  Processed {z+1}/{d} slices")
    
    kidney_pixels = np.sum(prediction)
    total_pixels = prediction.size
    coverage = (kidney_pixels / total_pixels) * 100
    
    print(f"✅ Prediction complete: {kidney_pixels} kidney pixels ({coverage:.2f}%)")
    
    return prediction

if __name__ == "__main__":
    model_file = train_kidney_model()
    
    if model_file:
        print(f"\n✅ SUCCESS! Model ready for integration.")
        print(f"📁 Model file: {model_file}")
        print(f"🔗 Next: Integrate with kidney_segmentation_pipeline.py")
    else:
        print("❌ Training failed!")
