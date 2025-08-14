"""
Region-Based Random Forest Kidney Detection
Extracts features from entire kidney regions, not individual voxels
"""
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
from datetime import datetime
from scipy.ndimage import gaussian_filter, sobel, label, center_of_mass
from skimage.measure import regionprops
import warnings
warnings.filterwarnings('ignore')

class RegionBasedKidneyRandomForest:
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        """Initialize region-based Random Forest kidney detector"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_region_features(self, mri_volume, region_mask):
        """Extract features from a kidney region"""
        features = []
        
        # Get region voxels
        region_voxels = mri_volume[region_mask > 0]
        
        if len(region_voxels) == 0:
            return [0] * 25  # Return zeros if no region
        
        # 1. Intensity statistics
        features.extend([
            np.mean(region_voxels),
            np.std(region_voxels),
            np.min(region_voxels),
            np.max(region_voxels),
            np.median(region_voxels),
            np.percentile(region_voxels, 25),
            np.percentile(region_voxels, 75),
            np.percentile(region_voxels, 90),
            np.percentile(region_voxels, 10)
        ])
        
        # 2. Shape features
        region_size = np.sum(region_mask)
        z_coords, y_coords, x_coords = np.where(region_mask > 0)
        
        if len(z_coords) > 0:
            # Bounding box
            z_span = np.max(z_coords) - np.min(z_coords) + 1
            y_span = np.max(y_coords) - np.min(y_coords) + 1
            x_span = np.max(x_coords) - np.min(x_coords) + 1
            
            # Center of mass
            com_z, com_y, com_x = center_of_mass(region_mask)
            
            # Normalized position (relative to volume center)
            vol_z, vol_y, vol_x = mri_volume.shape
            norm_com_z = com_z / vol_z
            norm_com_y = com_y / vol_y
            norm_com_x = com_x / vol_x
            
            features.extend([
                region_size,
                z_span, y_span, x_span,
                z_span * y_span * x_span,  # bounding box volume
                region_size / (z_span * y_span * x_span),  # compactness
                norm_com_z, norm_com_y, norm_com_x
            ])
        else:
            features.extend([0] * 9)
        
        # 3. Texture features (simplified)
        if len(region_voxels) > 1:
            # Gradient within region
            grad_z = sobel(mri_volume, axis=0)[region_mask > 0]
            grad_y = sobel(mri_volume, axis=1)[region_mask > 0]
            grad_x = sobel(mri_volume, axis=2)[region_mask > 0]
            grad_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
            
            features.extend([
                np.mean(grad_magnitude),
                np.std(grad_magnitude),
                np.max(grad_magnitude)
            ])
        else:
            features.extend([0, 0, 0])
        
        # 4. Context features (surrounding tissue)
        # Dilate region to get surrounding context
        from scipy.ndimage import binary_dilation
        dilated_mask = binary_dilation(region_mask, iterations=3)
        context_mask = dilated_mask & ~region_mask
        
        if np.sum(context_mask) > 0:
            context_voxels = mri_volume[context_mask]
            features.extend([
                np.mean(context_voxels),
                np.std(context_voxels),
                np.mean(region_voxels) - np.mean(context_voxels)  # contrast
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def generate_candidate_regions(self, mri_volume, min_size=1000, max_size=50000):
        """Generate candidate kidney regions using various methods"""
        print(f"    Generating candidate regions...")
        
        # Normalize volume
        volume_norm = (mri_volume - mri_volume.min()) / (mri_volume.max() - mri_volume.min())
        
        candidates = []
        
        # Method 1: Intensity thresholding
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            binary_mask = volume_norm > threshold
            labeled_mask, num_labels = label(binary_mask)
            
            for i in range(1, num_labels + 1):
                region_mask = (labeled_mask == i)
                region_size = np.sum(region_mask)
                
                if min_size <= region_size <= max_size:
                    candidates.append({
                        'mask': region_mask,
                        'size': region_size,
                        'method': f'threshold_{threshold}',
                        'is_kidney': False  # Default, will be set based on overlap
                    })
        
        # Method 2: Gradient-based regions
        grad_magnitude = np.sqrt(
            sobel(volume_norm, axis=0)**2 + 
            sobel(volume_norm, axis=1)**2 + 
            sobel(volume_norm, axis=2)**2
        )
        
        for threshold in [0.1, 0.2, 0.3]:
            binary_mask = grad_magnitude > threshold
            labeled_mask, num_labels = label(binary_mask)
            
            for i in range(1, num_labels + 1):
                region_mask = (labeled_mask == i)
                region_size = np.sum(region_mask)
                
                if min_size <= region_size <= max_size:
                    candidates.append({
                        'mask': region_mask,
                        'size': region_size,
                        'method': f'gradient_{threshold}',
                        'is_kidney': False
                    })
        
        print(f"      Generated {len(candidates)} candidate regions")
        return candidates
    
    def prepare_training_data(self, data_file='kidney_training_data.pkl'):
        """Prepare region-based training data"""
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
        
        all_features = []
        all_labels = []
        
        # Set feature names
        self.feature_names = [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'median_intensity',
            'p25_intensity', 'p75_intensity', 'p90_intensity', 'p10_intensity',
            'region_size', 'z_span', 'y_span', 'x_span', 'bbox_volume', 'compactness',
            'com_z_norm', 'com_y_norm', 'com_x_norm',
            'grad_mean', 'grad_std', 'grad_max',
            'context_mean', 'context_std', 'contrast'
        ]
        
        for vol_idx, (mri_vol, kidney_mask) in enumerate(zip(mri_volumes, kidney_masks)):
            print(f"   Processing volume {vol_idx+1}/{len(mri_volumes)}: {metadata[vol_idx]['file']}")
            
            # Generate candidate regions
            candidates = self.generate_candidate_regions(mri_vol)
            
            # Label candidates based on overlap with true kidney mask
            for candidate in candidates:
                # Calculate overlap with true kidney mask
                overlap = np.sum(candidate['mask'] & (kidney_mask > 0))
                overlap_ratio = overlap / candidate['size'] if candidate['size'] > 0 else 0
                
                # Consider it a kidney if significant overlap
                candidate['is_kidney'] = overlap_ratio > 0.5
            
            # Also add true kidney regions
            labeled_kidneys, num_kidneys = label(kidney_mask)
            for i in range(1, num_kidneys + 1):
                true_kidney_mask = (labeled_kidneys == i)
                if np.sum(true_kidney_mask) > 1000:  # Minimum size
                    candidates.append({
                        'mask': true_kidney_mask,
                        'size': np.sum(true_kidney_mask),
                        'method': 'true_kidney',
                        'is_kidney': True
                    })
            
            # Extract features for all candidates
            for candidate in candidates:
                features = self.extract_region_features(mri_vol, candidate['mask'])
                all_features.append(features)
                all_labels.append(1 if candidate['is_kidney'] else 0)
            
            n_kidney = sum(1 for c in candidates if c['is_kidney'])
            n_total = len(candidates)
            print(f"      Processed {n_total} regions ({n_kidney} kidney, {n_total-n_kidney} non-kidney)")
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n📊 Training data summary:")
        print(f"   Total regions: {len(X):,}")
        print(f"   Kidney regions: {np.sum(y):,} ({100*np.sum(y)/len(y):.1f}%)")
        print(f"   Features per region: {X.shape[1]}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """Train Random Forest on regions"""
        print(f"\n🌳 Training Region-Based Random Forest...")
        print(f"   Estimators: {self.n_estimators}")
        print(f"   Max depth: {self.max_depth}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"   Training regions: {X_train.shape[0]:,}")
        print(f"   Test regions: {X_test.shape[0]:,}")
        
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
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['Non-kidney', 'Kidney']))
        
        # Feature importance
        print("\n🔍 Top 10 Most Important Features:")
        feature_importance = self.model.feature_importances_
        importance_pairs = list(zip(self.feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(importance_pairs[:10]):
            print(f"   {i+1:2d}. {feature:20s}: {importance:.4f}")
        
        return {
            'train_f1': train_f1,
            'test_f1': test_f1,
            'training_time': training_time,
            'feature_importance': importance_pairs
        }
    
    def predict_kidneys(self, mri_volume):
        """Predict kidney regions in a new volume"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Generate candidate regions
        candidates = self.generate_candidate_regions(mri_volume)
        
        if not candidates:
            return np.zeros_like(mri_volume), []
        
        # Extract features for all candidates
        features_list = []
        for candidate in candidates:
            features = self.extract_region_features(mri_volume, candidate['mask'])
            features_list.append(features)
        
        # Scale and predict
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create output mask with predicted kidneys
        kidney_mask = np.zeros_like(mri_volume)
        detected_kidneys = []
        
        for i, (candidate, pred, prob) in enumerate(zip(candidates, predictions, probabilities)):
            if pred == 1 and prob > 0.5:  # Kidney prediction with confidence
                kidney_mask[candidate['mask']] = 1
                detected_kidneys.append({
                    'mask': candidate['mask'],
                    'confidence': prob,
                    'size': candidate['size'],
                    'method': candidate['method']
                })
        
        return kidney_mask, detected_kidneys
    
    def save_model(self, model_path='kidney_random_forest_regions.joblib'):
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
            'model_type': 'region_based'
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved: {model_path}")

def main():
    """Main training script"""
    print("🌳 REGION-BASED RANDOM FOREST KIDNEY DETECTION")
    print("="*50)
    
    # Configuration
    N_ESTIMATORS = 100
    MAX_DEPTH = 15
    
    # Initialize trainer
    trainer = RegionBasedKidneyRandomForest(
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
    model_path = f'kidney_random_forest_regions_{timestamp}.joblib'
    trainer.save_model(model_path)
    
    # Also save as best model
    trainer.save_model('kidney_random_forest_best.joblib')
    
    print(f"\n✅ Training complete!")
    print(f"   Best F1 Score: {results['test_f1']:.4f}")
    print(f"   Model saved: {model_path}")

if __name__ == "__main__":
    main()
