"""
AI Kidney Pipeline with .mat file output
Saves AI kidney detection results back to .mat format
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

# Import the U-Net model from our pipeline
from ai_kidney_pipeline_final import KidneyUNet3D, AIKidneySegmentationPipeline

def save_kidney_results_to_mat(original_mat_file, kidney_mask, bounding_boxes, output_filename=None):
    """Save AI kidney detection results to .mat file - Simple format for MATLAB combination"""
    
    if output_filename is None:
        # Create output filename
        input_path = Path(original_mat_file)
        output_filename = f"{input_path.stem}_AI_results.mat"
    
    print(f"💾 Saving AI kidney results to: {output_filename}")
    
    try:
        # Create simplified AI results structure (MATLAB-compatible)
        output_data = {}
        
        # Add AI kidney detection results 
        output_data['ai_kidney_mask'] = kidney_mask.astype(np.uint8)
        output_data['ai_detection_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Simple model info 
        output_data['ai_model_type'] = 'UNet3D'
        output_data['ai_training_f1_score'] = 0.836
        output_data['ai_training_iou'] = 0.718
        
        # Convert bounding boxes to MATLAB-style arrays
        if bounding_boxes:
            num_kidneys = len(bounding_boxes)
            output_data['ai_num_kidneys_detected'] = num_kidneys
            
            # Create simple arrays for bounding box data
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
        
        # Add summary statistics
        total_kidney_voxels = int(np.sum(kidney_mask))
        total_voxels = int(kidney_mask.size)
        coverage_percent = float((total_kidney_voxels / total_voxels) * 100)
        
        output_data['ai_total_kidney_voxels'] = total_kidney_voxels
        output_data['ai_total_voxels'] = total_voxels
        output_data['ai_coverage_percent'] = coverage_percent
        
        # Save AI results only (simple structure)
        print("   💾 Saving AI results .mat file...")
        scipy.io.savemat(output_filename, output_data, format='5', do_compression=True)
        
        print(f"✅ AI results saved successfully!")
        print(f"   File: {output_filename}")
        print(f"   Kidneys detected: {len(bounding_boxes) if bounding_boxes else 0}")
        print(f"   Coverage: {coverage_percent:.2f}%")
        print(f"   � Use MATLAB script to combine with original Arbuz file")
        
        return output_filename
        
    except Exception as e:
        print(f"❌ Error saving .mat file: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_and_save_kidney_detection(mat_file, output_dir=None):
    """Complete pipeline: process MRI and save results to .mat file"""
    
    print(f"\n🔄 PROCESSING: {Path(mat_file).name}")
    print("="*60)
    
    # Initialize AI pipeline
    try:
        pipeline = AIKidneySegmentationPipeline()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    
    # Process the file
    result = pipeline.process_mri_file(mat_file, visualize=True)
    
    if result is None:
        print("❌ Failed to process MRI file")
        return None
    
    # Create timestamped output directory in alignProcess/data/inference
    if output_dir is None:
        # Navigate to alignProcess/data/inference
        current_dir = Path.cwd()
        # Go up from src/creatingKidneys/src to alignProcess
        alignprocess_dir = current_dir.parent.parent.parent
        inference_base_dir = alignprocess_dir / "data" / "inference"
        
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = inference_base_dir / f"kidney_detection_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created output directory: {output_dir}")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    input_name = Path(mat_file).stem
    output_filename = output_dir / f"{input_name}_with_AI_kidneys.mat"
    
    # Save results to .mat file
    saved_file = save_kidney_results_to_mat(
        mat_file, 
        result['kidney_mask'], 
        result['bounding_boxes'],
        str(output_filename)
    )
    
    # Also copy/move the visualization to the same directory
    if saved_file:
        # Look for the generated visualization
        viz_pattern = f"ai_kidney_detection_{input_name}.png"
        viz_files = list(Path(".").glob(viz_pattern))
        
        if viz_files:
            viz_source = viz_files[0]
            viz_dest = output_dir / viz_source.name
            
            try:
                import shutil
                shutil.copy2(viz_source, viz_dest)
                print(f"📊 Copied visualization: {viz_dest}")
            except Exception as e:
                print(f"⚠️  Could not copy visualization: {e}")
    
    return saved_file, output_dir

def main():
    """Main function to process all .mat files and save AI results"""
    print("AI KIDNEY DETECTION WITH .MAT OUTPUT")
    print("="*50)
    
    # Find .mat files to process
    mat_files = []
    
    # Look in current directory
    mat_files.extend(Path(".").glob("*.mat"))
    
    if not mat_files:
        print("No .mat files found in current directory!")
        return
    
    print(f"📁 Found {len(mat_files)} .mat files to process")
    
    # Process each file
    processed_files = []
    output_directories = set()
    
    for mat_file in mat_files:
        result = process_and_save_kidney_detection(str(mat_file))
        if result and len(result) == 2:
            saved_file, output_dir = result
            if saved_file:
                processed_files.append(saved_file)
                output_directories.add(output_dir)
    
    # Summary
    print(f"\n📋 PROCESSING COMPLETE")
    print("="*30)
    print(f"   Files processed: {len(processed_files)}")
    
    if output_directories:
        print(f"   Output directories created:")
        for output_dir in output_directories:
            print(f"     📁 {output_dir}")
            
            # List contents of each directory
            try:
                contents = list(output_dir.glob("*"))
                for item in contents:
                    if item.suffix == '.mat':
                        print(f"       📄 {item.name}")
                    elif item.suffix == '.png':
                        print(f"       🖼️  {item.name}")
            except Exception as e:
                print(f"       ⚠️  Could not list contents: {e}")
    
    print(f"\n🎉 AI kidney detection with organized .mat output complete!")
    
    # Show what's in the output files
    if processed_files:
        print(f"\n🔍 OUTPUT .MAT FILE CONTENTS:")
        example_file = processed_files[0]
        try:
            data = scipy.io.loadmat(example_file)
            ai_keys = [k for k in data.keys() if k.startswith('ai_')]
            print(f"   AI-added keys: {ai_keys}")
            
            if 'ai_num_kidneys_detected' in data:
                num_kidneys = data['ai_num_kidneys_detected'][0, 0]
                print(f"   Kidneys detected: {num_kidneys}")
            
            if 'ai_kidney_stats' in data:
                stats = data['ai_kidney_stats'][0, 0]
                if hasattr(stats, 'dtype'):
                    print(f"   Kidney statistics available")
                
        except Exception as e:
            print(f"   Could not read example file: {e}")
    
    # Final directory structure info
    if output_directories:
        print(f"\n📍 FILES SAVED TO:")
        for output_dir in output_directories:
            print(f"   {output_dir}")
            relative_path = output_dir.relative_to(Path.cwd().parent.parent.parent)
            print(f"   (Relative: {relative_path})")

if __name__ == "__main__":
    main()
