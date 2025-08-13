#!/usr/bin/env python3
"""
Enhanced Kidney Visualization - Using existing successful AI results
Creates clear visualizations showing kidneys on MRI background
"""

import sys
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class KidneyVisualizationEnhancer:
    """Enhanced visualization using existing AI results"""
    
    def create_enhanced_kidney_visualization(self, mri_data, ai_results_file, output_path):
        """Create comprehensive kidney detection visualization"""
        try:
            print("Creating enhanced visualization with kidney overlays...")
            
            # Load AI results
            ai_data = scipy.io.loadmat(ai_results_file)
            
            # Get kidney mask and bounding box info
            kidney_mask = ai_data['ai_kidney_mask']
            num_kidneys = int(ai_data['ai_num_kidneys_detected'])
            
            print(f"Found {num_kidneys} kidneys in AI results")
            
            # Get bounding box information
            bounding_boxes = []
            if num_kidneys > 0:
                for i in range(num_kidneys):
                    box = {
                        'center_y': float(ai_data['ai_center_y'][0, i]),
                        'center_x': float(ai_data['ai_center_x'][0, i]),
                        'center_z': float(ai_data['ai_center_z'][0, i]),
                        'min_y': float(ai_data['ai_bounds_min_y'][0, i]),
                        'max_y': float(ai_data['ai_bounds_max_y'][0, i]),
                        'min_x': float(ai_data['ai_bounds_min_x'][0, i]),
                        'max_x': float(ai_data['ai_bounds_max_x'][0, i]),
                        'min_z': float(ai_data['ai_bounds_min_z'][0, i]),
                        'max_z': float(ai_data['ai_bounds_max_z'][0, i]),
                        'size': float(ai_data['ai_kidney_sizes'][0, i])
                    }
                    bounding_boxes.append(box)
                    print(f"  Kidney {i+1}: center ({box['center_y']:.0f}, {box['center_x']:.0f}, {box['center_z']:.0f})")
            
            # Normalize MRI for display
            mri_display = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
            
            # Create comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle('Enhanced AI Kidney Detection Results', fontsize=18, fontweight='bold', y=0.95)
            
            # Choose slices that show kidneys best
            z_center = int(mri_data.shape[2] // 2)
            kidney_slices = []
            
            # Find slices with kidney pixels
            for z in range(mri_data.shape[2]):
                if np.sum(kidney_mask[:, :, z]) > 0:
                    kidney_slices.append(z)
            
            # Select representative slices
            if len(kidney_slices) >= 5:
                selected_slices = [
                    kidney_slices[0],  # First slice with kidneys
                    kidney_slices[len(kidney_slices)//4],
                    kidney_slices[len(kidney_slices)//2],  # Middle
                    kidney_slices[3*len(kidney_slices)//4],
                    kidney_slices[-1]  # Last slice with kidneys
                ]
            else:
                # Default slices if not enough kidney slices
                selected_slices = [
                    max(0, z_center - 6),
                    max(0, z_center - 3),
                    z_center,
                    min(mri_data.shape[2] - 1, z_center + 3),
                    min(mri_data.shape[2] - 1, z_center + 6)
                ]
            
            print(f"Showing slices: {selected_slices}")
            
            # Row 1: MRI with kidney overlays
            for i, z_slice in enumerate(selected_slices):
                ax = plt.subplot(4, 5, i + 1)
                
                # Show MRI background
                ax.imshow(mri_display[:, :, z_slice], cmap='gray', alpha=0.9)
                
                # Overlay kidney mask
                mask_slice = kidney_mask[:, :, z_slice]
                if np.sum(mask_slice) > 0:
                    ax.imshow(mask_slice, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
                
                # Add bounding boxes for kidneys in this slice
                colors = ['yellow', 'cyan', 'lime', 'magenta']
                for j, box in enumerate(bounding_boxes):
                    if box['min_z'] <= z_slice <= box['max_z']:
                        rect = patches.Rectangle(
                            (box['min_x'], box['min_y']), 
                            box['max_x'] - box['min_x'], 
                            box['max_y'] - box['min_y'],
                            linewidth=2, edgecolor=colors[j % len(colors)], 
                            facecolor='none', linestyle='--'
                        )
                        ax.add_patch(rect)
                        
                        # Add kidney label
                        ax.text(box['min_x'], box['min_y'] - 3, f'K{j+1}', 
                               color=colors[j % len(colors)], fontweight='bold', fontsize=10)
                
                ax.set_title(f'MRI + Kidneys (Z={z_slice})', fontsize=11, fontweight='bold')
                ax.axis('off')
            
            # Row 2: Kidney masks only
            for i, z_slice in enumerate(selected_slices):
                ax = plt.subplot(4, 5, i + 6)
                mask_slice = kidney_mask[:, :, z_slice]
                
                if np.sum(mask_slice) > 0:
                    ax.imshow(mask_slice, cmap='Reds', vmin=0, vmax=1)
                else:
                    # Show empty slice
                    ax.imshow(np.zeros_like(mask_slice), cmap='gray', vmin=0, vmax=1)
                
                ax.set_title(f'Kidney Mask (Z={z_slice})', fontsize=10)
                ax.axis('off')
            
            # Row 3: Original MRI only
            for i, z_slice in enumerate(selected_slices):
                ax = plt.subplot(4, 5, i + 11)
                ax.imshow(mri_display[:, :, z_slice], cmap='gray')
                ax.set_title(f'Original MRI (Z={z_slice})', fontsize=10)
                ax.axis('off')
            
            # Row 4: Summary information and 3D projection
            
            # Best slice summary
            ax_summary = plt.subplot(4, 5, 16)
            # Find slice with most kidney pixels
            kidney_counts = [np.sum(kidney_mask[:, :, z]) for z in range(kidney_mask.shape[2])]
            best_slice = np.argmax(kidney_counts)
            
            ax_summary.imshow(mri_display[:, :, best_slice], cmap='gray')
            mask_best = kidney_mask[:, :, best_slice]
            ax_summary.imshow(mask_best, cmap='Reds', alpha=0.5)
            
            # Add all bounding boxes on best slice
            for j, box in enumerate(bounding_boxes):
                if box['min_z'] <= best_slice <= box['max_z']:
                    rect = patches.Rectangle(
                        (box['min_x'], box['min_y']), 
                        box['max_x'] - box['min_x'], 
                        box['max_y'] - box['min_y'],
                        linewidth=3, edgecolor='yellow', facecolor='none'
                    )
                    ax_summary.add_patch(rect)
                    ax_summary.text(box['min_x'], box['min_y'] - 5, f'Kidney {j+1}', 
                                   color='yellow', fontweight='bold', fontsize=12)
            
            ax_summary.set_title(f'Best View (Z={best_slice})', fontsize=12, fontweight='bold')
            ax_summary.axis('off')
            
            # Statistics text
            ax_stats = plt.subplot(4, 5, 17)
            ax_stats.axis('off')
            
            stats_text = f"AI KIDNEY DETECTION SUMMARY\n\n"
            stats_text += f"Kidneys Detected: {num_kidneys}\n"
            stats_text += f"Total Kidney Voxels: {np.sum(kidney_mask):,}\n"
            stats_text += f"MRI Volume: {mri_data.shape}\n"
            stats_text += f"Best Slice: {best_slice}\n"
            coverage = (np.sum(kidney_mask) / kidney_mask.size) * 100
            stats_text += f"Kidney Coverage: {coverage:.2f}%\n\n"
            
            if 'ai_training_f1_score' in ai_data:
                stats_text += f"Model F1 Score: {float(ai_data['ai_training_f1_score']):.3f}\n"
            if 'ai_training_iou' in ai_data:
                stats_text += f"Model IoU: {float(ai_data['ai_training_iou']):.3f}\n"
            
            stats_text += f"\nProcessed: {ai_data['ai_detection_timestamp'][0]}"
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            # Individual kidney details
            ax_kidneys = plt.subplot(4, 5, 18)
            ax_kidneys.axis('off')
            
            kidney_text = "KIDNEY DETAILS\n\n"
            for i, box in enumerate(bounding_boxes):
                kidney_text += f"Kidney {i+1}:\n"
                kidney_text += f"  Size: {box['size']:,.0f} voxels\n"
                kidney_text += f"  Center: ({box['center_y']:.0f}, {box['center_x']:.0f}, {box['center_z']:.0f})\n"
                kidney_text += f"  Z-range: {box['min_z']:.0f} - {box['max_z']:.0f}\n"
                kidney_text += f"  Coverage: {(box['size']/kidney_mask.size)*100:.2f}%\n\n"
            
            ax_kidneys.text(0.05, 0.95, kidney_text, transform=ax_kidneys.transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
            
            # 3D visualization (maximum intensity projection)
            ax_3d = plt.subplot(4, 5, 19)
            
            # MIP of MRI
            mri_mip = np.max(mri_display, axis=2)
            ax_3d.imshow(mri_mip, cmap='gray', alpha=0.8)
            
            # MIP of kidney mask
            kidney_mip = np.max(kidney_mask, axis=2)
            ax_3d.imshow(kidney_mip, cmap='Reds', alpha=0.6)
            
            ax_3d.set_title('3D Projection (MIP)', fontsize=10, fontweight='bold')
            ax_3d.axis('off')
            
            # Color legend
            ax_legend = plt.subplot(4, 5, 20)
            ax_legend.axis('off')
            
            # Create color patches for legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.6, label='Kidney Tissue'),
                plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.8, label='MRI Background'),
                plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='yellow', linewidth=2, label='Bounding Box')
            ]
            
            ax_legend.legend(handles=legend_elements, loc='center', fontsize=10)
            ax_legend.set_title('Legend', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Make room for main title
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Enhanced visualization saved: {Path(output_path).name}")
            
            # Create a simple summary view
            self.create_simple_summary(mri_data, kidney_mask, bounding_boxes, 
                                     output_path.replace('.png', '_summary.png'))
            
        except Exception as e:
            print(f"Error creating enhanced visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def create_simple_summary(self, mri_data, kidney_mask, bounding_boxes, output_path):
        """Create a simple 3-panel summary view"""
        try:
            # Normalize MRI for display
            mri_display = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
            
            # Find best slice (slice with most kidney pixels)
            kidney_counts = [np.sum(kidney_mask[:, :, z]) for z in range(kidney_mask.shape[2])]
            best_slice = np.argmax(kidney_counts)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Panel 1: Original MRI
            axes[0].imshow(mri_display[:, :, best_slice], cmap='gray')
            axes[0].set_title('Original MRI', fontsize=16, fontweight='bold', pad=20)
            axes[0].axis('off')
            
            # Panel 2: MRI + Kidney overlay with bounding boxes
            axes[1].imshow(mri_display[:, :, best_slice], cmap='gray', alpha=0.8)
            mask_slice = kidney_mask[:, :, best_slice]
            axes[1].imshow(mask_slice, cmap='Reds', alpha=0.6)
            
            # Add bounding boxes and labels
            colors = ['yellow', 'cyan']
            for i, box in enumerate(bounding_boxes):
                if box['min_z'] <= best_slice <= box['max_z']:
                    rect = patches.Rectangle(
                        (box['min_x'], box['min_y']), 
                        box['max_x'] - box['min_x'], 
                        box['max_y'] - box['min_y'],
                        linewidth=4, edgecolor=colors[i % len(colors)], facecolor='none'
                    )
                    axes[1].add_patch(rect)
                    axes[1].text(box['min_x'], box['min_y'] - 8, f'Kidney {i+1}', 
                               color=colors[i % len(colors)], fontweight='bold', fontsize=14,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
            
            axes[1].set_title(f'AI Detection (Slice {best_slice})', fontsize=16, fontweight='bold', pad=20)
            axes[1].axis('off')
            
            # Panel 3: Kidney segmentation only
            axes[2].imshow(mask_slice, cmap='Reds')
            axes[2].set_title('Kidney Segmentation', fontsize=16, fontweight='bold', pad=20)
            axes[2].axis('off')
            
            # Add detection summary at bottom
            summary_text = f"🎯 Kidneys Detected: {len(bounding_boxes)}   📊 Coverage: {(np.sum(kidney_mask)/kidney_mask.size)*100:.1f}%   🎬 Best Slice: {best_slice}"
            
            fig.text(0.5, 0.02, summary_text, ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Summary visualization saved: {Path(output_path).name}")
            
        except Exception as e:
            print(f"Error creating summary visualization: {e}")

def main():
    """Main function for enhanced kidney visualization"""
    if len(sys.argv) < 4:
        print("Usage: python enhanced_viz.py original_mri.mat ai_results.mat output_directory")
        return
    
    original_file = sys.argv[1]
    ai_results_file = sys.argv[2] 
    output_dir = sys.argv[3]
    
    print("ENHANCED KIDNEY VISUALIZATION")
    print("="*40)
    
    if not os.path.exists(original_file):
        print(f"Error: Original file not found: {original_file}")
        return
        
    if not os.path.exists(ai_results_file):
        print(f"Error: AI results file not found: {ai_results_file}")
        return
    
    try:
        # Load original MRI data
        print(f"Loading MRI data from: {os.path.basename(original_file)}")
        mat_data = scipy.io.loadmat(original_file)
        
        # Find MRI data in 'images' structure
        mri_data = None
        if 'images' in mat_data:
            images = mat_data['images']
            print(f"   Images structure shape: {images.shape}")
            
            for i in range(images.shape[1]):  # Shape is (1, 6)
                try:
                    img = images[0, i]  # Access as (0, i)
                    
                    if hasattr(img, 'dtype') and 'data' in img.dtype.names:
                        # Access the data field
                        data = img['data'][0, 0]
                        
                        if hasattr(data, 'shape') and len(data.shape) >= 3:
                            print(f"   Found MRI data in images[0,{i}]: {data.shape}")
                            print(f"   Value range: [{np.min(data):.3f}, {np.max(data):.3f}]")
                            mri_data = data
                            break
                    elif hasattr(img, 'shape') and len(img.shape) >= 3:
                        print(f"   Found direct MRI data in images[0,{i}]: {img.shape}")
                        mri_data = img
                        break
                        
                except Exception as e:
                    print(f"   Could not access images[0,{i}]: {e}")
                    continue
        
        if mri_data is None:
            print("Could not find valid MRI data in file")
            return
        
        # Create enhanced visualizations
        visualizer = KidneyVisualizationEnhancer()
        
        input_name = Path(original_file).stem
        output_path = Path(output_dir) / f"ENHANCED_kidney_visualization_{input_name}.png"
        
        visualizer.create_enhanced_kidney_visualization(mri_data, ai_results_file, str(output_path))
        
        print("\nENHANCED VISUALIZATION COMPLETE!")
        print("="*40)
        print("✅ Kidneys are now clearly visible on MRI background")
        print("✅ Bounding boxes show kidney locations")
        print("✅ Multiple slice views provided")
        print("✅ Summary statistics included")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
