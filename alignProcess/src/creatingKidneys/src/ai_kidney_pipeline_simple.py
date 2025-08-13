#!/usr/bin/env python3
"""
AI Kidney Segmentation Pipeline - Simple ASCII Version
Creates Arbuz-compatible .mat files for kidney detection
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class KidneyUNet3D(nn.Module):
    """3D U-Net for kidney segmentation"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(KidneyUNet3D, self).__init__()
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Down part of UNET
        for feature in features:
            self.downs.append(self._double_conv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self._double_conv(feature*2, feature))
        
        self.bottleneck = self._double_conv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))

class AIKidneySegmentationPipeline:
    """Complete AI pipeline for kidney segmentation"""
    
    def __init__(self, model_path):
        print(f"Loading trained model: {model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = KidneyUNet3D()
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"   Training F1 Score: {checkpoint.get('f1_score', 'Unknown')}")
                print(f"   Training IoU: {checkpoint.get('iou', 'Unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"   Warning: Model file not found, using random weights")
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def preprocess_mri(self, mri_data):
        """Preprocess MRI data for AI model"""
        # Normalize to [0, 1]
        mri_normalized = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data) + 1e-8)
        
        # Add batch and channel dimensions
        mri_tensor = torch.FloatTensor(mri_normalized).unsqueeze(0).unsqueeze(0)
        return mri_tensor.to(self.device)
    
    def detect_kidneys(self, mri_data):
        """Run AI kidney detection on MRI data"""
        print("Running AI kidney segmentation...")
        
        with torch.no_grad():
            # Preprocess
            mri_tensor = self.preprocess_mri(mri_data)
            
            # Run inference
            prediction = self.model(mri_tensor)
            
            # Convert to numpy
            kidney_mask = prediction.cpu().numpy().squeeze()
            
            # Apply threshold
            kidney_mask = (kidney_mask > 0.5).astype(np.uint8)
            
        return kidney_mask
    
    def analyze_kidneys(self, kidney_mask):
        """Analyze kidney segmentation to find individual kidneys"""
        from scipy import ndimage
        
        # Find connected components
        labeled_mask, num_features = ndimage.label(kidney_mask)
        
        print(f"Found {num_features} connected components")
        
        bounding_boxes = []
        
        for i in range(1, num_features + 1):
            # Get component mask
            component_mask = (labeled_mask == i)
            component_size = np.sum(component_mask)
            
            # Skip very small components
            if component_size < 1000:
                continue
            
            # Find bounding box
            coords = np.where(component_mask)
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            min_z, max_z = np.min(coords[2]), np.max(coords[2])
            
            # Calculate center
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            center_z = (min_z + max_z) // 2
            
            print(f"  Kidney {i}: {component_size} voxels, center ({center_y}, {center_x}, {center_z})")
            
            bounding_boxes.append({
                'size': component_size,
                'bounds': (min_y, max_y, min_x, max_x, min_z, max_z),
                'center': (center_y, center_x, center_z)
            })
        
        return bounding_boxes
    
    def process_mat_file(self, mat_file_path):
        """Process a .mat file for kidney detection"""
        print(f"\nAI KIDNEY SEGMENTATION: {Path(mat_file_path).name}")
        print("="*60)
        
        try:
            # Load .mat file
            mat_data = scipy.io.loadmat(mat_file_path)
            
            # Find MRI data in 'images' structure
            print("Searching in 'images' structure...")
            if 'images' in mat_data:
                images = mat_data['images']
                
                # Handle different image structures
                mri_data = None
                for i in range(len(images)):
                    try:
                        if hasattr(images[i], 'shape') and len(images[i].shape) >= 2:
                            data = images[i]
                        else:
                            # Try accessing data field
                            data = images[i]['data'][0, 0] if 'data' in images[i].dtype.names else images[i][0, 0]
                        
                        if hasattr(data, 'shape') and len(data.shape) >= 3:
                            print(f"   Found MRI data: {data.shape}")
                            mri_data = data
                            print(f"Found MRI data: images[{i}].data {data.shape}")
                            print(f"   Value range: [{np.min(data):.3f}, {np.max(data):.3f}]")
                            break
                    except:
                        continue
                
                if mri_data is None:
                    print("Could not find valid MRI data in images structure")
                    return None, None
                
                # Run AI detection
                kidney_mask = self.detect_kidneys(mri_data)
                
                # Analyze results
                bounding_boxes = self.analyze_kidneys(kidney_mask)
                
                if bounding_boxes:
                    print(f"AI detected {len(bounding_boxes)} kidney region(s):")
                    total_voxels = 0
                    for i, box in enumerate(bounding_boxes):
                        print(f"   Region {i+1}: bounds {box['bounds']}, {box['size']} voxels ({box['size']/mri_data.size*100:.2f}%)")
                        total_voxels += box['size']
                    
                    print(f"Total kidney coverage: {total_voxels/mri_data.size*100:.2f}%")
                else:
                    print("No significant kidney regions detected")
                
                return kidney_mask, bounding_boxes
                
            else:
                print("No 'images' structure found in .mat file")
                return None, None
                
        except Exception as e:
            print(f"Error processing {mat_file_path}: {e}")
            return None, None
    
    def save_visualization(self, kidney_mask, bounding_boxes, output_path):
        """Save visualization of kidney detection"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Show different slices
            z_slices = [kidney_mask.shape[2]//4, kidney_mask.shape[2]//2, 
                       3*kidney_mask.shape[2]//4, -1]
            
            for i, z in enumerate(z_slices):
                ax = axes[i//2, i%2]
                ax.imshow(kidney_mask[:, :, z], cmap='Reds', alpha=0.7)
                ax.set_title(f'Kidney Mask - Slice {z}')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"AI visualization saved: {Path(output_path).name}")
            
        except Exception as e:
            print(f"Error saving visualization: {e}")

def save_ai_results_to_mat(original_mat_file, kidney_mask, bounding_boxes, output_filename):
    """Save AI kidney detection results to .mat file"""
    
    print(f"Saving AI kidney results to: {output_filename}")
    
    try:
        # Create simplified AI results structure
        output_data = {}
        
        # Add AI kidney detection results 
        output_data['ai_kidney_mask'] = kidney_mask.astype(np.uint8)
        output_data['ai_detection_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Model info 
        output_data['ai_model_type'] = 'UNet3D'
        output_data['ai_training_f1_score'] = 0.836
        output_data['ai_training_iou'] = 0.718
        
        # Bounding boxes
        if bounding_boxes:
            num_kidneys = len(bounding_boxes)
            output_data['ai_num_kidneys_detected'] = num_kidneys
            
            # Create arrays for bounding box data
            output_data['ai_kidney_ids'] = np.array([i+1 for i in range(num_kidneys)], dtype=np.float64)
            output_data['ai_kidney_sizes'] = np.array([box['size'] for box in bounding_boxes], dtype=np.float64)
            
            # Bounds arrays
            output_data['ai_bounds_min_y'] = np.array([box['bounds'][0] for box in bounding_boxes], dtype=np.float64)
            output_data['ai_bounds_max_y'] = np.array([box['bounds'][1] for box in bounding_boxes], dtype=np.float64)
            output_data['ai_bounds_min_x'] = np.array([box['bounds'][2] for box in bounding_boxes], dtype=np.float64)
            output_data['ai_bounds_max_x'] = np.array([box['bounds'][3] for box in bounding_boxes], dtype=np.float64)
            output_data['ai_bounds_min_z'] = np.array([box['bounds'][4] for box in bounding_boxes], dtype=np.float64)
            output_data['ai_bounds_max_z'] = np.array([box['bounds'][5] for box in bounding_boxes], dtype=np.float64)
            
            # Center arrays
            output_data['ai_center_y'] = np.array([box['center'][0] for box in bounding_boxes], dtype=np.float64)
            output_data['ai_center_x'] = np.array([box['center'][1] for box in bounding_boxes], dtype=np.float64)
            output_data['ai_center_z'] = np.array([box['center'][2] for box in bounding_boxes], dtype=np.float64)
        else:
            output_data['ai_num_kidneys_detected'] = 0
        
        # Summary statistics
        total_kidney_voxels = int(np.sum(kidney_mask))
        total_voxels = int(kidney_mask.size)
        coverage_percent = float((total_kidney_voxels / total_voxels) * 100)
        
        output_data['ai_total_kidney_voxels'] = total_kidney_voxels
        output_data['ai_total_voxels'] = total_voxels
        output_data['ai_coverage_percent'] = coverage_percent
        
        # Save AI results
        print("   Saving AI results .mat file...")
        scipy.io.savemat(output_filename, output_data, format='5', do_compression=True)
        
        print(f"AI results saved successfully!")
        print(f"   File: {output_filename}")
        print(f"   Kidneys detected: {len(bounding_boxes) if bounding_boxes else 0}")
        print(f"   Coverage: {coverage_percent:.2f}%")
        print(f"   Use MATLAB script to combine with original Arbuz file")
        
        return output_filename
        
    except Exception as e:
        print(f"Error saving .mat file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function - handles single file processing"""
    if len(sys.argv) < 3:
        print("Usage: python script.py input_file.mat output_directory")
        return
    
    input_file = sys.argv[1]
    output_base_dir = sys.argv[2]
    
    print("AI KIDNEY DETECTION WITH .MAT OUTPUT")
    print("="*50)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print(f"Processing: {os.path.basename(input_file)}")
    
    # Initialize pipeline
    pipeline = AIKidneySegmentationPipeline("kidney_unet_model_best.pth")
    
    # Process file
    kidney_mask, bounding_boxes = pipeline.process_mat_file(input_file)
    
    if kidney_mask is not None:
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = Path(input_file).stem
        output_dir = Path(output_base_dir) / f"kidney_detection_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Created output directory: {output_dir}")
        
        # Save AI results
        output_filename = output_dir / f"{input_name}_AI_results.mat"
        save_ai_results_to_mat(input_file, kidney_mask, bounding_boxes, str(output_filename))
        
        # Save visualization
        viz_file = output_dir / f"ai_kidney_detection_{input_name}.png"
        pipeline.save_visualization(kidney_mask, bounding_boxes, str(viz_file))
        
        print("\nPROCESSING COMPLETE")
        print("="*30)
        print(f"Output directory: {output_dir}")
        print("Files created:")
        for file in output_dir.glob("*"):
            if file.is_file():
                print(f"  {file.name}")
        
        print("\nNext step: Use MATLAB to create Arbuz-compatible file:")
        print(f"combine_arbuz_with_ai('{input_file}', '{output_filename}', 'output_ARBUZ_COMPATIBLE.mat')")
    
    else:
        print("Failed to process file")

if __name__ == "__main__":
    main()
