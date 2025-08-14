"""
Random Forest Kidney Detection Model - Region-Based
Compatible with existing AI detection pipeline
Works on entire kidney regions instead of individual voxels
"""
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, sobel, label, center_of_mass, binary_dilation
import warnings
warnings.filterwarnings('ignore')

class RandomForestKidneyModel:
    """Region-based Random Forest model compatible with existing AI pipeline"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metadata = {}
        
        if model_path:
            self.load_model(model_path)
    
    def extract_region_features(self, mri_volume, region_mask):
        """Extract features from a kidney region"""
        features = []
        
        # Get region voxels
        region_voxels = mri_volume[region_mask > 0]
        
        if len(region_voxels) == 0:
            return [0] * 24  # Return zeros if no region
        
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
        
        # 3. Texture features
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
        
        # 4. Context features
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
        """Generate candidate kidney regions"""
        # Normalize volume
        volume_norm = (mri_volume - mri_volume.min()) / (mri_volume.max() - mri_volume.min())
        
        candidates = []
        
        # Intensity thresholding
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
                        'method': f'threshold_{threshold}'
                    })
        
        return candidates
    
    def load_model(self, model_path):
        """Load trained Random Forest model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            
            # Load metadata if available
            metadata_path = model_path.replace('.joblib', '_metadata.pkl')
            try:
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            except:
                self.metadata = {}
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def forward(self, x):
        """Forward pass - compatible with PyTorch-style interface"""
        # Convert to numpy if needed
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        
        # Remove batch dimension if present
        if len(x.shape) == 5:  # [batch, channel, depth, height, width]
            x = x[0, 0]  # Take first sample, first channel
        elif len(x.shape) == 4:  # [channel, depth, height, width]
            x = x[0]  # Take first channel
        
        # Generate candidate regions
        candidates = self.generate_candidate_regions(x)
        
        if not candidates:
            # Return zeros if no candidates found
            result = np.zeros_like(x)[np.newaxis, np.newaxis, :, :, :]
            return result
        
        # Extract features for all candidates
        features_list = []
        for candidate in candidates:
            features = self.extract_region_features(x, candidate['mask'])
            features_list.append(features)
        
        # Scale and predict
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create output mask with predicted kidneys
        kidney_mask = np.zeros_like(x)
        
        for i, (candidate, pred, prob) in enumerate(zip(candidates, predictions, probabilities)):
            if pred == 1 and prob > 0.3:  # Lower confidence threshold
                kidney_mask[candidate['mask']] = prob  # Use probability as intensity
        
        # Add batch and channel dimensions to match PyTorch format
        result = kidney_mask[np.newaxis, np.newaxis, :, :, :]  # [1, 1, D, H, W]
        
        return result
    
    def predict(self, mri_volume):
        """Predict kidney mask for new MRI volume"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Generate candidate regions
        candidates = self.generate_candidate_regions(mri_volume)
        
        if not candidates:
            return np.zeros_like(mri_volume), np.zeros_like(mri_volume, dtype=np.float32)
        
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
        
        # Create output masks
        prediction_volume = np.zeros_like(mri_volume)
        probability_volume = np.zeros_like(mri_volume, dtype=np.float32)
        
        for i, (candidate, pred, prob) in enumerate(zip(candidates, predictions, probabilities)):
            if pred == 1 and prob > 0.3:  # Lower confidence threshold
                prediction_volume[candidate['mask']] = 1
                probability_volume[candidate['mask']] = prob
        
        return prediction_volume, probability_volume
    
    def eval(self):
        """Set model to evaluation mode (for compatibility)"""
        pass
    
    def to(self, device):
        """Device transfer (for compatibility with PyTorch interface)"""
        return self

# Create a wrapper class that matches the expected interface
class RandomForestKidneyDetector:
    """Wrapper to make Random Forest compatible with existing pipeline"""
    
    def __init__(self, model_path='kidney_random_forest_best.joblib'):
        self.model = RandomForestKidneyModel(model_path)
        self.device = 'cpu'  # Random Forest runs on CPU
    
    def load_state_dict(self, state_dict):
        """For compatibility - Random Forest doesn't use state_dict"""
        pass
    
    def eval(self):
        """Set to evaluation mode"""
        self.model.eval()
        return self
    
    def to(self, device):
        """Device transfer"""
        self.device = device
        return self.model.to(device)
    
    def forward(self, x):
        """Forward pass"""
        return self.model.forward(x)
    
    def __call__(self, x):
        """Make callable"""
        return self.forward(x)

def load_random_forest_model(model_path='kidney_random_forest_best.joblib'):
    """Load Random Forest model with PyTorch-compatible interface"""
    try:
        model = RandomForestKidneyDetector(model_path)
        print(f"✅ Random Forest model loaded from {model_path}")
        
        # Print model info if available
        if hasattr(model.model, 'metadata') and model.model.metadata:
            metadata = model.model.metadata
            print(f"   Model type: {metadata.get('model_type', 'RandomForest')}")
            print(f"   Features: {metadata.get('n_features', 'unknown')}")
            print(f"   Trained: {metadata.get('training_timestamp', 'unknown')}")
        
        return model
    except Exception as e:
        print(f"❌ Error loading Random Forest model: {e}")
        return None
