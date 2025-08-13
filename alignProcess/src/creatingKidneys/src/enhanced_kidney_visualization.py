#!/usr/bin/env python3
"""
Enhanced AI Kidney Visualization Pipeline
Creates clear visualizations showing kidneys on MRI background
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class KidneyUNet3D(nn.Module):
    """3D U-Net for kidney segmentation - matches training architecture"""
    def __init__(self, in_channels=1, out_channels=1):
        super(KidneyUNet3D, self).__init__()
        
        # Encoder
        self.enc1 = self._double_conv(in_channels, 32)
        self.enc2 = self._double_conv(32, 64)  
        self.enc3 = self._double_conv(64, 128)
        
        # Bottleneck
        self.bottleneck = self._double_conv(128, 128)
        
        # Decoder  
        self.up3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(64, 32)
        
        # Final
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool3d(2)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),  
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1) 
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.final(dec1))

class EnhancedKidneyVisualization:
    """Enhanced kidney detection with superior visualization"""
    
    def __init__(self, model_path):
        print(f"Loading trained model: {model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model with correct architecture
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
                'center': (center_y, center_x, center_z),
                'label': i
            })
        
        return bounding_boxes, labeled_mask
    
    def create_enhanced_visualization(self, mri_data, kidney_mask, bounding_boxes, labeled_mask, output_path):
        """Create comprehensive kidney detection visualization"""
        try:
            print("Creating enhanced visualization...")
            
            # Normalize MRI for display
            mri_display = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
            
            # Create figure with multiple views
            fig = plt.figure(figsize=(20, 15))
            
            # Choose representative slices
            z_center = mri_data.shape[2] // 2
            slices_to_show = [
                max(0, z_center - 4),
                z_center - 2, 
                z_center,
                z_center + 2,
                min(mri_data.shape[2] - 1, z_center + 4)
            ]
            
            # Main title
            fig.suptitle('AI Kidney Detection Results', fontsize=16, fontweight='bold')
            
            # Create subplots for different views
            for i, z_slice in enumerate(slices_to_show):
                # Original MRI + Kidney overlay
                ax1 = plt.subplot(3, 5, i + 1)
                ax1.imshow(mri_display[:, :, z_slice], cmap='gray', alpha=0.8)
                
                # Overlay kidney mask with different colors for each kidney
                mask_slice = kidney_mask[:, :, z_slice]
                labeled_slice = labeled_mask[:, :, z_slice]
                
                # Color each kidney differently
                colors = ['red', 'blue', 'green', 'yellow', 'purple']
                for box in bounding_boxes:
                    kidney_id = box['label']
                    kidney_pixels = (labeled_slice == kidney_id)
                    if np.any(kidney_pixels):
                        color = colors[(kidney_id - 1) % len(colors)]
                        ax1.imshow(kidney_pixels, cmap='Reds' if color == 'red' else 'Blues' if color == 'blue' else 'Greens', 
                                  alpha=0.4, vmin=0, vmax=1)
                        
                        # Add bounding box if kidney is visible in this slice
                        if (box['bounds'][4] <= z_slice <= box['bounds'][5]):
                            rect = patches.Rectangle(
                                (box['bounds'][2], box['bounds'][0]), 
                                box['bounds'][3] - box['bounds'][2], 
                                box['bounds'][1] - box['bounds'][0],
                                linewidth=2, edgecolor=color, facecolor='none'
                            )
                            ax1.add_patch(rect)
                            
                            # Add kidney label
                            ax1.text(box['bounds'][2], box['bounds'][0] - 5, f'K{kidney_id}', 
                                   color=color, fontweight='bold', fontsize=12)
                
                ax1.set_title(f'Slice {z_slice} (Overlay)', fontsize=10)
                ax1.axis('off')
                
                # Kidney mask only
                ax2 = plt.subplot(3, 5, i + 6)
                ax2.imshow(mask_slice, cmap='Reds', alpha=0.8)
                ax2.set_title(f'Mask Only - Slice {z_slice}', fontsize=10)
                ax2.axis('off')
                
                # Original MRI only
                ax3 = plt.subplot(3, 5, i + 11)
                ax3.imshow(mri_display[:, :, z_slice], cmap='gray')
                ax3.set_title(f'Original MRI - Slice {z_slice}', fontsize=10)
                ax3.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Enhanced visualization saved: {Path(output_path).name}")
            
            # Create summary visualization
            self.create_summary_visualization(mri_data, kidney_mask, bounding_boxes, 
                                            output_path.replace('.png', '_summary.png'))
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def create_summary_visualization(self, mri_data, kidney_mask, bounding_boxes, output_path):
        """Create a single summary view"""
        try:
            # Normalize MRI for display
            mri_display = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
            
            # Find best slice (slice with most kidney pixels)
            kidney_counts = [np.sum(kidney_mask[:, :, z]) for z in range(kidney_mask.shape[2])]
            best_slice = np.argmax(kidney_counts)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original MRI
            axes[0].imshow(mri_display[:, :, best_slice], cmap='gray')
            axes[0].set_title('Original MRI', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # MRI + Kidney overlay
            axes[1].imshow(mri_display[:, :, best_slice], cmap='gray')
            mask_slice = kidney_mask[:, :, best_slice]
            axes[1].imshow(mask_slice, cmap='Reds', alpha=0.5)
            
            # Add bounding boxes and labels
            for i, box in enumerate(bounding_boxes):
                if box['bounds'][4] <= best_slice <= box['bounds'][5]:
                    rect = patches.Rectangle(
                        (box['bounds'][2], box['bounds'][0]), 
                        box['bounds'][3] - box['bounds'][2], 
                        box['bounds'][1] - box['bounds'][0],
                        linewidth=3, edgecolor='yellow', facecolor='none'
                    )
                    axes[1].add_patch(rect)
                    axes[1].text(box['bounds'][2], box['bounds'][0] - 5, f'Kidney {i+1}', 
                               color='yellow', fontweight='bold', fontsize=12)
            
            axes[1].set_title(f'AI Detection (Slice {best_slice})', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Kidney mask only
            axes[2].imshow(mask_slice, cmap='Reds')
            axes[2].set_title('Kidney Segmentation', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # Add detection summary
            summary_text = f"Kidneys Detected: {len(bounding_boxes)}\n"
            summary_text += f"Best Slice: {best_slice}\n"
            total_pixels = np.sum(kidney_mask)
            coverage = (total_pixels / kidney_mask.size) * 100
            summary_text += f"Coverage: {coverage:.1f}%"
            
            fig.text(0.02, 0.02, summary_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Summary visualization saved: {Path(output_path).name}")
            
        except Exception as e:
            print(f"Error creating summary visualization: {e}")
    
    def process_mat_file(self, mat_file_path, output_dir=None):
        """Process a .mat file and create enhanced visualizations"""
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
                bounding_boxes, labeled_mask = self.analyze_kidneys(kidney_mask)
                
                if bounding_boxes:
                    print(f"AI detected {len(bounding_boxes)} kidney region(s):")
                    total_voxels = 0
                    for i, box in enumerate(bounding_boxes):
                        print(f"   Region {i+1}: bounds {box['bounds']}, {box['size']} voxels ({box['size']/mri_data.size*100:.2f}%)")
                        total_voxels += box['size']
                    
                    print(f"Total kidney coverage: {total_voxels/mri_data.size*100:.2f}%")
                    
                    # Create enhanced visualizations
                    if output_dir:
                        input_name = Path(mat_file_path).stem
                        viz_path = Path(output_dir) / f"enhanced_kidney_detection_{input_name}.png"
                        self.create_enhanced_visualization(mri_data, kidney_mask, bounding_boxes, 
                                                         labeled_mask, str(viz_path))
                else:
                    print("No significant kidney regions detected")
                
                return kidney_mask, bounding_boxes, mri_data
                
            else:
                print("No 'images' structure found in .mat file")
                return None, None, None
                
        except Exception as e:
            print(f"Error processing {mat_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

def main():
    """Main function for enhanced kidney visualization"""
    if len(sys.argv) < 2:
        print("Usage: python enhanced_kidney_viz.py input_file.mat [output_directory]")
        return
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    print("ENHANCED AI KIDNEY DETECTION & VISUALIZATION")
    print("="*50)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print(f"Processing: {os.path.basename(input_file)}")
    
    # Initialize enhanced pipeline
    pipeline = EnhancedKidneyVisualization("kidney_unet_model_best.pth")
    
    # Process file with enhanced visualization
    kidney_mask, bounding_boxes, mri_data = pipeline.process_mat_file(input_file, output_dir)
    
    if kidney_mask is not None:
        print("\nENHANCED VISUALIZATION COMPLETE")
        print("="*35)
        print("Created files:")
        print(f"  - Enhanced multi-slice view")
        print(f"  - Summary view with bounding boxes")
        print(f"  - Kidneys clearly visible on MRI background")
        print("\nKidney masks and boundaries are now clearly visible!")
    else:
        print("Failed to process file")

if __name__ == "__main__":
    main()
