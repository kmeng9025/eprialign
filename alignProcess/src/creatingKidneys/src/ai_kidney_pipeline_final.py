"""
AI-Powered Kidney Segmentation Pipeline
Replaces static boxes with trained U-Net model
"""
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from scipy.ndimage import zoom
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Define the same U-Net architecture used in training
class KidneyUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(KidneyUNet3D, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Encoder (same as training)
        self.enc1 = conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(64, 128)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)
        
        # Final layer
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, apply_sigmoid=True):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        logits = self.final(dec1)
        
        if apply_sigmoid:
            return self.sigmoid(logits)
        else:
            return logits

class AIKidneySegmentationPipeline:
    def __init__(self, model_path=None):
        """Initialize the AI kidney segmentation pipeline"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.target_size = (64, 64, 32)  # Model training size
        
        # Load the trained model
        if model_path is None:
            # Look for the best model
            model_files = [
                "kidney_unet_model_best.pth",
                "kidney_unet_model.pth"
            ]
            
            for model_file in model_files:
                if Path(model_file).exists():
                    model_path = model_file
                    break
            
            if model_path is None:
                raise FileNotFoundError("No trained kidney model found! Run train_kidney_unet.py first.")
        
        self.load_model(model_path)
        print(f"🧠 AI Kidney Pipeline initialized with model: {model_path}")
    
    def load_model(self, model_path):
        """Load the trained U-Net model"""
        print(f"📂 Loading trained model: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            if 'target_size' in checkpoint:
                self.target_size = checkpoint['target_size']
            
            # Print training info if available
            if 'val_metrics' in checkpoint:
                metrics = checkpoint['val_metrics']
                print(f"   Training F1 Score: {metrics.get('f1', 'N/A'):.3f}")
                print(f"   Training IoU: {metrics.get('iou', 'N/A'):.3f}")
        else:
            model_state_dict = checkpoint
        
        # Initialize and load model
        self.model = KidneyUNet3D().to(self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def resize_volume(self, volume, target_shape):
        """Resize 3D volume to target shape"""
        if volume.shape == target_shape:
            return volume
        
        zoom_factors = [t/s for t, s in zip(target_shape, volume.shape)]
        return zoom(volume, zoom_factors, order=1)
    
    def predict_kidney_mask(self, mri_volume, threshold=0.5):
        """Predict kidney mask using the trained U-Net"""
        original_shape = mri_volume.shape
        
        # Normalize MRI data
        mri_normalized = mri_volume.astype(np.float32)
        if np.max(mri_normalized) > 1.0:
            mri_normalized = mri_normalized / np.max(mri_normalized)
        
        # Resize to model input size
        mri_resized = self.resize_volume(mri_normalized, self.target_size)
        
        # Add batch and channel dimensions: (1, 1, D, H, W)
        mri_tensor = torch.tensor(mri_resized[np.newaxis, np.newaxis, ...], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(mri_tensor, apply_sigmoid=True)
            prediction_np = prediction.cpu().numpy()[0, 0]  # Remove batch and channel dims
        
        # Apply threshold
        binary_mask = (prediction_np > threshold).astype(np.uint8)
        
        # Resize back to original size
        kidney_mask_original = self.resize_volume(binary_mask.astype(np.float32), original_shape)
        kidney_mask_original = (kidney_mask_original > 0.5).astype(np.uint8)
        
        return kidney_mask_original, prediction_np
    
    def extract_kidney_bounding_boxes(self, kidney_mask, min_size=100):
        """Extract bounding boxes for detected kidneys"""
        from scipy import ndimage
        
        # Find connected components
        labeled_array, num_features = ndimage.label(kidney_mask)
        
        print(f"🔍 Found {num_features} connected components")
        
        boxes = []
        for label in range(1, num_features + 1):
            component = (labeled_array == label)
            component_size = np.sum(component)
            
            if component_size < min_size:
                continue
            
            # Find bounding box
            coords = np.where(component)
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            min_z, max_z = np.min(coords[2]), np.max(coords[2])
            
            # Add some padding
            h, w, d = kidney_mask.shape
            padding = 5
            
            min_y = max(0, min_y - padding)
            max_y = min(h - 1, max_y + padding)
            min_x = max(0, min_x - padding)
            max_x = min(w - 1, max_x + padding)
            min_z = max(0, min_z - padding)
            max_z = min(d - 1, max_z + padding)
            
            box = {
                'size': component_size,
                'bounds': (min_y, max_y, min_x, max_x, min_z, max_z),
                'center': ((min_y + max_y) // 2, (min_x + max_x) // 2, (min_z + max_z) // 2),
                'label': label
            }
            
            boxes.append(box)
            print(f"  Kidney {label}: {component_size} voxels, center {box['center']}")
        
        # Sort by size (largest first)
        boxes.sort(key=lambda x: x['size'], reverse=True)
        
        return boxes
    
    def process_mri_file(self, mat_file, visualize=True):
        """Process a single MRI file with AI kidney detection"""
        print(f"\n🏥 AI KIDNEY SEGMENTATION: {Path(mat_file).name}")
        print("="*60)
        
        try:
            # Load the .mat file
            data = scipy.io.loadmat(mat_file)
            
            # Find MRI data using the discovered structure
            mri_data = None
            data_key = None
            
            # First try direct keys for simple structure
            for key, value in data.items():
                if key.startswith('_'):
                    continue
                
                if hasattr(value, 'shape') and len(value.shape) == 3:
                    if value.shape[0] == 350 and value.shape[1] == 350:  # MRI dimensions
                        mri_data = value
                        data_key = key
                        break
            
            # If not found, look in 'images' structure (withoutROIwithMRI.mat format)
            if mri_data is None and 'images' in data:
                print("🔍 Searching in 'images' structure...")
                
                images = data['images']
                if hasattr(images, 'flat'):
                    for i, img_item in enumerate(images.flat):
                        if hasattr(img_item, 'dtype') and 'data' in img_item.dtype.names:
                            img_data = img_item['data']
                            
                            # Handle (1,1) shaped data containers
                            if img_data.shape == (1, 1):
                                actual_data = img_data[0, 0]
                                if hasattr(actual_data, 'shape') and len(actual_data.shape) == 3:
                                    h, w, d = actual_data.shape
                                    if h == 350 and w == 350:  # MRI dimensions
                                        mri_data = actual_data
                                        data_key = f"images[{i}].data"
                                        print(f"   Found MRI data: {actual_data.shape}")
                                        break
                            elif hasattr(img_data, 'shape') and len(img_data.shape) == 3:
                                h, w, d = img_data.shape
                                if h == 350 and w == 350:  # MRI dimensions
                                    mri_data = img_data
                                    data_key = f"images[{i}].data"
                                    print(f"   Found MRI data: {img_data.shape}")
                                    break
            
            if mri_data is None:
                print("❌ No MRI data found in file")
                print("   Available keys:", [k for k in data.keys() if not k.startswith('_')])
                return None
            
            print(f"📊 Found MRI data: {data_key} {mri_data.shape}")
            print(f"   Value range: [{np.min(mri_data):.3f}, {np.max(mri_data):.3f}]")
            
            # AI Prediction
            print("🤖 Running AI kidney segmentation...")
            kidney_mask, prediction_confidence = self.predict_kidney_mask(mri_data)
            
            # Extract bounding boxes
            boxes = self.extract_kidney_bounding_boxes(kidney_mask)
            
            if len(boxes) == 0:
                print("⚠️  No kidneys detected!")
                # Create fallback boxes (center of image)
                h, w, d = mri_data.shape
                fallback_box = {
                    'size': 0,
                    'bounds': (h//4, 3*h//4, w//4, 3*w//4, d//4, 3*d//4),
                    'center': (h//2, w//2, d//2),
                    'label': 'fallback'
                }
                boxes = [fallback_box]
                print("📦 Using fallback bounding box")
            
            print(f"🎯 AI detected {len(boxes)} kidney region(s):")
            total_kidney_voxels = np.sum(kidney_mask)
            coverage = (total_kidney_voxels / kidney_mask.size) * 100
            
            for i, box in enumerate(boxes):
                bounds = box['bounds']
                size = box['size']
                box_coverage = (size / kidney_mask.size) * 100 if size > 0 else 0
                print(f"   Region {i+1}: bounds {bounds}, {size} voxels ({box_coverage:.2f}%)")
            
            print(f"📊 Total kidney coverage: {coverage:.2f}%")
            
            # Create visualization
            if visualize:
                self.create_kidney_visualization(mri_data, kidney_mask, boxes, mat_file)
            
            result = {
                'file': mat_file,
                'mri_shape': mri_data.shape,
                'kidney_mask': kidney_mask,
                'prediction_confidence': prediction_confidence,
                'bounding_boxes': boxes,
                'total_kidney_voxels': total_kidney_voxels,
                'coverage_percent': coverage,
                'method': 'AI_UNet'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {mat_file}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_kidney_visualization(self, mri_data, kidney_mask, boxes, filename):
        """Create visualization of AI kidney detection results"""
        
        # Choose middle slice for visualization
        h, w, d = mri_data.shape
        mid_slice = d // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original MRI
        axes[0, 0].imshow(mri_data[:, :, mid_slice], cmap='gray')
        axes[0, 0].set_title(f'Original MRI (slice {mid_slice})')
        axes[0, 0].axis('off')
        
        # AI Kidney mask overlay
        axes[0, 1].imshow(mri_data[:, :, mid_slice], cmap='gray')
        mask_overlay = np.ma.masked_where(kidney_mask[:, :, mid_slice] == 0, kidney_mask[:, :, mid_slice])
        axes[0, 1].imshow(mask_overlay, alpha=0.6, cmap='Reds')
        axes[0, 1].set_title('AI Kidney Detection')
        axes[0, 1].axis('off')
        
        # Bounding boxes
        axes[0, 2].imshow(mri_data[:, :, mid_slice], cmap='gray')
        
        # Draw bounding boxes that intersect this slice
        colors = ['red', 'blue', 'green', 'yellow', 'cyan']
        for i, box in enumerate(boxes):
            min_y, max_y, min_x, max_x, min_z, max_z = box['bounds']
            
            if min_z <= mid_slice <= max_z:
                # Draw rectangle
                from matplotlib.patches import Rectangle
                color = colors[i % len(colors)]
                rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                               linewidth=3, edgecolor=color, facecolor='none')
                axes[0, 2].add_patch(rect)
                
                # Add label
                axes[0, 2].text(min_x, min_y - 5, f'AI Kidney {i+1}', 
                            color=color, fontsize=12, weight='bold')
        
        axes[0, 2].set_title('AI Bounding Boxes')
        axes[0, 2].axis('off')
        
        # 3D visualizations
        # Sagittal view (middle)
        mid_x = w // 2
        axes[1, 0].imshow(mri_data[:, mid_x, :].T, cmap='gray', aspect='auto')
        mask_sag = np.ma.masked_where(kidney_mask[:, mid_x, :].T == 0, kidney_mask[:, mid_x, :].T)
        axes[1, 0].imshow(mask_sag, alpha=0.6, cmap='Reds', aspect='auto')
        axes[1, 0].set_title(f'Sagittal View (x={mid_x})')
        axes[1, 0].axis('off')
        
        # Coronal view (middle)
        mid_y = h // 2
        axes[1, 1].imshow(mri_data[mid_y, :, :].T, cmap='gray', aspect='auto')
        mask_cor = np.ma.masked_where(kidney_mask[mid_y, :, :].T == 0, kidney_mask[mid_y, :, :].T)
        axes[1, 1].imshow(mask_cor, alpha=0.6, cmap='Reds', aspect='auto')
        axes[1, 1].set_title(f'Coronal View (y={mid_y})')
        axes[1, 1].axis('off')
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""AI Kidney Segmentation Results
        
File: {Path(filename).name}
MRI Shape: {mri_data.shape}
        
AI Detection:
• Total kidney voxels: {np.sum(kidney_mask):,}
• Coverage: {(np.sum(kidney_mask)/kidney_mask.size)*100:.2f}%
• Regions found: {len(boxes)}

Model: U-Net 3D
Device: {self.device}
Target size: {self.target_size}"""
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save visualization
        output_name = f"ai_kidney_detection_{Path(filename).stem}.png"
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 AI visualization saved: {output_name}")

def main():
    """Main AI kidney segmentation pipeline"""
    print("🚀 AI KIDNEY SEGMENTATION PIPELINE")
    print("🤖 Using Trained U-Net Model")
    print("="*50)
    
    # Initialize AI pipeline
    try:
        pipeline = AIKidneySegmentationPipeline()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Find test files
    test_files = []
    
    # Look for .mat files in data directory
    data_dir = Path("../../data")
    if data_dir.exists():
        test_files.extend(data_dir.glob("*.mat"))
    
    # Look in current directory
    test_files.extend(Path(".").glob("*.mat"))
    
    if not test_files:
        print("❌ No .mat files found for testing!")
        print("   Place .mat files in current directory or ../../data/")
        return
    
    print(f"📁 Found {len(test_files)} test files")
    
    # Process each file
    results = []
    
    for mat_file in test_files[:3]:  # Process first 3 files
        result = pipeline.process_mri_file(str(mat_file), visualize=True)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n📋 AI KIDNEY SEGMENTATION SUMMARY")
    print("="*40)
    print(f"   Files processed: {len(results)}")
    
    for result in results:
        print(f"   {Path(result['file']).name}:")
        print(f"     Shape: {result['mri_shape']}")
        print(f"     AI detected kidneys: {len(result['bounding_boxes'])}")
        print(f"     Coverage: {result['coverage_percent']:.2f}%")
        print(f"     Method: {result['method']}")
    
    print(f"\n🎉 AI kidney segmentation complete!")
    print(f"   ✅ Static boxes successfully replaced with AI model!")
    print(f"   🧠 Using trained U-Net for intelligent kidney detection")

if __name__ == "__main__":
    main()
