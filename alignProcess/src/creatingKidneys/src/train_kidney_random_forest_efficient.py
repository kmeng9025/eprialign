"""
Memory-Efficient Random Forest Kidney Detection Training
Uses sampling and patch-based approach to handle large datasets
"""
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
from datetime import datetime
from scipy.ndimage import gaussian_filter, sobel
import warnings
warnings.filterwarnings('ignore')

class EfficientKidneyRandomForest:
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        """Initialize efficient Random Forest kidney detector"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_patch_features(self, mri_patch):
        """Extract features from a small MRI patch"""
        # Simple but effective features for patches
        features = []
        
        # 1. Intensity statistics
        features.extend([
            np.mean(mri_patch),
            np.std(mri_patch),
            np.min(mri_patch),
            np.max(mri_patch),
            np.median(mri_patch)
        ])
        
        # 2. Percentiles
        features.extend([
            np.percentile(mri_patch, 25),
            np.percentile(mri_patch, 75)
        ])
        
        # 3. Gradient information
        if mri_patch.size > 1:
            grad = np.gradient(mri_patch.flatten())
            features.extend([
                np.mean(np.abs(grad)),
                np.std(grad)
            ])
        else:
            features.extend([0, 0])
        
        return features
    
    def sample_training_patches(self, mri_volumes, kidney_masks, 
                              patch_size=5, max_samples_per_volume=10000,
                              kidney_ratio=0.3):
        """Sample patches from volumes for training"""
        print("🔬 Sampling training patches...")
        
        all_features = []
        all_labels = []
        
        half_patch = patch_size // 2
        
        for vol_idx, (mri_vol, kidney_mask) in enumerate(zip(mri_volumes, kidney_masks)):
            print(f"   Volume {vol_idx+1}/{len(mri_volumes)}: shape {mri_vol.shape}")
            
            depth, height, width = mri_vol.shape
            
            # Find kidney and non-kidney voxel indices
            kidney_indices = np.where(kidney_mask == 1)
            non_kidney_indices = np.where(kidney_mask == 0)
            
            kidney_coords = list(zip(kidney_indices[0], kidney_indices[1], kidney_indices[2]))
            non_kidney_coords = list(zip(non_kidney_indices[0], non_kidney_indices[1], non_kidney_indices[2]))
            
            # Sample kidney patches
            n_kidney_samples = min(
                int(max_samples_per_volume * kidney_ratio),
                len(kidney_coords)
            )
            
            if n_kidney_samples > 0:
                kidney_sample_idx = np.random.choice(
                    len(kidney_coords), n_kidney_samples, replace=False
                )
                
                for idx in kidney_sample_idx:
                    z, y, x = kidney_coords[idx]
                    
                    # Extract patch
                    z_min = max(0, z - half_patch)
                    z_max = min(depth, z + half_patch + 1)
                    y_min = max(0, y - half_patch)
                    y_max = min(height, y + half_patch + 1)
                    x_min = max(0, x - half_patch)
                    x_max = min(width, x + half_patch + 1)
                    
                    patch = mri_vol[z_min:z_max, y_min:y_max, x_min:x_max]
                    
                    # Extract features
                    features = self.extract_patch_features(patch)
                    
                    # Add position features (normalized)
                    features.extend([z/depth, y/height, x/width])
                    
                    all_features.append(features)
                    all_labels.append(1)
            
            # Sample non-kidney patches
            n_non_kidney_samples = max_samples_per_volume - n_kidney_samples
            
            if n_non_kidney_samples > 0 and len(non_kidney_coords) > 0:
                non_kidney_sample_idx = np.random.choice(
                    len(non_kidney_coords), 
                    min(n_non_kidney_samples, len(non_kidney_coords)), 
                    replace=False
                )
                
                for idx in non_kidney_sample_idx:
                    z, y, x = non_kidney_coords[idx]
                    
                    # Extract patch
                    z_min = max(0, z - half_patch)
                    z_max = min(depth, z + half_patch + 1)
                    y_min = max(0, y - half_patch)
                    y_max = min(height, y + half_patch + 1)
                    x_min = max(0, x - half_patch)
                    x_max = min(width, x + half_patch + 1)
                    
                    patch = mri_vol[z_min:z_max, y_min:y_max, x_min:x_max]
                    
                    # Extract features
                    features = self.extract_patch_features(patch)
                    
                    # Add position features (normalized)
                    features.extend([z/depth, y/height, x/width])
                    
                    all_features.append(features)
                    all_labels.append(0)
            
            n_samples = len([f for f in all_features if len(f) > 0])
            n_kidney = len([l for l in all_labels if l == 1])
            print(f"      Sampled {n_samples} patches ({n_kidney} kidney, {n_samples-n_kidney} non-kidney)")
        
        # Set feature names
        self.feature_names = [
            'mean', 'std', 'min', 'max', 'median',
            'p25', 'p75', 'grad_mean', 'grad_std',
            'pos_z', 'pos_y', 'pos_x'
        ]
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"   Total samples: {len(X):,}")
        print(f"   Kidney samples: {np.sum(y):,} ({100*np.sum(y)/len(y):.1f}%)")
        print(f"   Features: {X.shape[1]}")
        
        return X, y
    
    def prepare_training_data(self, data_file='kidney_training_data.pkl'):
        """Load and prepare training data efficiently"""
        print("📦 Loading training data...")
        
        if not os.path.exists(data_file):
            print(f"❌ Training data file not found: {data_file}")
            return None, None
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        mri_volumes = data['mri_volumes']
        kidney_masks = data['kidney_masks'] 
        metadata = data['metadata']
        
        print(f"   Found {len(mri_volumes)} training volumes")
        
        # Sample patches instead of using all voxels
        X, y = self.sample_training_patches(mri_volumes, kidney_masks)
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """Train Random Forest model"""
        print(f"\n🌳 Training Efficient Random Forest...")
        print(f"   Estimators: {self.n_estimators}")
        print(f"   Max depth: {self.max_depth}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"   Training samples: {X_train.shape[0]:,}")
        print(f"   Test samples: {X_test.shape[0]:,}")
        
        # Scale features
        print("   🔧 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("   🌱 Training Random Forest...")
        start_time = time.time()
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        print(f"   ✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        
        # Training predictions
        y_train_pred = self.model.predict(X_train_scaled)
        train_f1 = f1_score(y_train, y_train_pred)
        
        # Test predictions
        y_test_pred = self.model.predict(X_test_scaled)
        test_f1 = f1_score(y_test, y_test_pred)
        
        print(f"   Training F1 Score: {train_f1:.4f}")
        print(f"   Test F1 Score: {test_f1:.4f}")
        
        # Detailed classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['Non-kidney', 'Kidney']))
        
        # Feature importance
        print("\n🔍 Feature Importance:")
        feature_importance = self.model.feature_importances_
        importance_pairs = list(zip(self.feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(importance_pairs):
            print(f"   {i+1:2d}. {feature:15s}: {importance:.4f}")
        
        return {
            'train_f1': train_f1,
            'test_f1': test_f1,
            'training_time': training_time,
            'feature_importance': importance_pairs
        }
    
    def save_model(self, model_path='kidney_random_forest_efficient.joblib'):
        """Save trained model"""
        if self.model is None:
            print("❌ No model to save!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'model_type': 'patch_based'
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Also save metadata
        metadata = {
            'model_type': 'RandomForest_PatchBased',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_timestamp': datetime.now().isoformat(),
            'patch_based': True
        }
        
        with open(model_path.replace('.joblib', '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
    
    def predict_volume(self, mri_volume, patch_size=5):
        """Predict kidney mask for full volume using sliding window"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        depth, height, width = mri_volume.shape
        prediction_volume = np.zeros_like(mri_volume)
        probability_volume = np.zeros_like(mri_volume, dtype=np.float32)
        
        half_patch = patch_size // 2
        
        # Sliding window prediction
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    # Extract patch
                    z_min = max(0, z - half_patch)
                    z_max = min(depth, z + half_patch + 1)
                    y_min = max(0, y - half_patch)
                    y_max = min(height, y + half_patch + 1)
                    x_min = max(0, x - half_patch)
                    x_max = min(width, x + half_patch + 1)
                    
                    patch = mri_volume[z_min:z_max, y_min:y_max, x_min:x_max]
                    
                    # Extract features
                    features = self.extract_patch_features(patch)
                    features.extend([z/depth, y/height, x/width])
                    
                    # Scale and predict
                    features_scaled = self.scaler.transform([features])
                    prediction = self.model.predict(features_scaled)[0]
                    probability = self.model.predict_proba(features_scaled)[0, 1]
                    
                    prediction_volume[z, y, x] = prediction
                    probability_volume[z, y, x] = probability
        
        return prediction_volume, probability_volume

def main():
    """Main training script"""
    print("🌳 EFFICIENT RANDOM FOREST KIDNEY DETECTION TRAINING")
    print("="*55)
    
    # Configuration - smaller for efficiency
    N_ESTIMATORS = 100  # Fewer trees
    MAX_DEPTH = 15      # Shallower trees
    
    # Initialize trainer
    trainer = EfficientKidneyRandomForest(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    
    # Prepare data
    X, y = trainer.prepare_training_data()
    if X is None:
        return
    
    # Train model
    results = trainer.train(X, y)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'kidney_random_forest_efficient_{timestamp}.joblib'
    trainer.save_model(model_path)
    
    # Also save best model
    trainer.save_model('kidney_random_forest_best.joblib')
    
    print(f"\n✅ Training complete!")
    print(f"   Best F1 Score: {results['test_f1']:.4f}")
    print(f"   Model saved: {model_path}")
    print(f"   Ready for inference!")

if __name__ == "__main__":
    main()
