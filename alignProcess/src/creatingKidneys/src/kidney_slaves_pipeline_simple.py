#!/usr/bin/env python3
"""
Kidney Slaves Pipeline - Creates kidney slaves in Arbuz projects
===============================================================

This pipeline:
1. Processes MRI data for kidney detection
2. Applies trained AI model
3. Creates temporary AI results
4. Calls MATLAB to create kidney slaves
5. Cleans up temporary files

The kidneys will appear as slaves in ArbuzGUI, just like the training files!

Author: AI Assistant
Date: 2025-08-13
"""

import torch
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom, label
import os
import sys
import subprocess
from datetime import datetime
import tempfile

# Import model architecture
sys.path.append(r'C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src')
from unet_3d import UNet3D

class KidneySlavesPipeline:
    """Kidney detection pipeline that creates proper Arbuz slaves"""
    
    def __init__(self, model_path):
        print("🚀 Initializing Kidney Slaves Pipeline...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   🔧 Device: {self.device}")
        
        # Load trained model
        self.model = self._load_model(model_path)
        print("   ✅ AI model loaded successfully")
        print("   🔧 MATLAB integration ready")
    
    def _load_model(self, model_path):
        """Load the trained kidney detection model"""
        print(f"   📂 Loading model: {model_path}")
        
        # Initialize model architecture
        model = UNet3D(in_channels=1, out_channels=1)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def process_file(self, input_file, output_dir=None):
        """
        Process a single .mat file and create final output with kidney slaves
        
        Args:
            input_file: Path to input .mat file
            output_dir: Output directory (optional, defaults to inference folder)
        
        Returns:
            str: Path to final .mat file with kidney slaves
        """
        print(f"\n🔄 Processing: {os.path.basename(input_file)}")
        
        # Set up output directory
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(base_dir, "data", "inference", f"kidney_slaves_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"   📁 Output directory: {output_dir}")
        
        try:
            # Step 1: Load and process MRI data
            print("   📂 Loading MRI data...")
            mri_data = self._load_mri_data(input_file)
            print(f"   ✅ MRI data loaded: {mri_data.shape}")
            
            # Step 2: Run AI kidney detection
            print("   🤖 Running AI kidney detection...")
            kidney_mask, confidence = self._detect_kidneys(mri_data)
            num_kidneys = self._count_kidneys(kidney_mask)
            print(f"   🎯 Detection complete: {num_kidneys} kidneys found (confidence: {confidence:.3f})")
            
            # Step 3: Create temporary AI results file
            temp_ai_file = os.path.join(output_dir, "temp_ai_results.mat")
            self._save_ai_results(temp_ai_file, kidney_mask, num_kidneys, confidence)
            
            # Step 4: Create final file with kidney slaves using MATLAB
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            final_output_file = os.path.join(output_dir, f"{base_name}_FINAL_KIDNEY_SLAVES.mat")
            
            print("   🎯 Creating kidney slaves in Arbuz project...")
            self._call_matlab_for_slaves(input_file, temp_ai_file, final_output_file)
            
            # Verify final file exists
            if os.path.exists(final_output_file):
                file_size_mb = os.path.getsize(final_output_file) / (1024*1024)
                print(f"\n✅ SUCCESS! Kidney slaves created:")
                print(f"   📁 File: {final_output_file}")
                print(f"   📊 Size: {file_size_mb:.1f} MB")
                print(f"   🎯 Kidney slaves: {num_kidneys}")
                print(f"   👁️  Kidneys will be visible as slaves in ArbuzGUI!")
                
                # Clean up temporary file
                try:
                    os.remove(temp_ai_file)
                    print(f"   🧹 Temporary files cleaned up")
                except:
                    pass
                
                return final_output_file
            else:
                raise FileNotFoundError("Final output file was not created")
                
        except Exception as e:
            print(f"❌ Error processing {input_file}: {str(e)}")
            raise
    
    def _call_matlab_for_slaves(self, input_file, ai_results_file, output_file):
        """Call MATLAB to create kidney slaves"""
        
        # Create MATLAB command
        matlab_cmd = [
            'matlab', '-batch',
            f"addpath('C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\src\\creatingKidneys\\src'); "
            f"create_kidney_slaves_final('{input_file}', '{ai_results_file}', '{output_file}'); "
            f"exit;"
        ]
        
        print(f"   🔧 Running MATLAB command...")
        
        try:
            # Run MATLAB command
            result = subprocess.run(matlab_cmd, capture_output=True, text=True, 
                                  cwd=os.path.dirname(__file__), encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                print(f"   ✅ MATLAB execution successful")
                # Print relevant output (filter out noise)
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if any(marker in line for marker in ['✅', '🎯', '📁', '📊', '🧹', '📋', 'kidney', 'slave']):
                            print(f"      {line}")
            else:
                print(f"   ❌ MATLAB error: {result.stderr}")
                raise RuntimeError(f"MATLAB execution failed: {result.stderr}")
                
        except FileNotFoundError:
            raise RuntimeError("MATLAB not found. Please ensure MATLAB is installed and in PATH.")
    
    def _load_mri_data(self, mat_file):
        """Load MRI data from Arbuz .mat file"""
        data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        
        # Look for MRI data in the images array (Arbuz structure)
        if 'images' in data:
            images = data['images']
            
            for i in range(len(images)):
                img = images[i]
                
                if img is None or (isinstance(img, np.ndarray) and img.size == 0):
                    continue
                
                # Look for MRI data (should be 3D)
                if (hasattr(img, 'data') and img.data is not None and 
                    hasattr(img.data, 'shape') and len(img.data.shape) == 3):
                    
                    mri_data = img.data
                    if min(mri_data.shape) > 1:  # Valid 3D data
                        print(f"   📊 Found MRI data in images[{i}]: {mri_data.shape}")
                        return np.array(mri_data, dtype=np.float32)
        
        # Fallback: look for any 3D array in top-level keys
        for key, value in data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray):
                if value.ndim >= 3 and min(value.shape) > 1:
                    print(f"   📊 Found MRI data in top-level key '{key}': {value.shape}")
                    return np.array(value, dtype=np.float32)
        
        raise ValueError("No suitable 3D MRI data found in file")
    
    def _detect_kidneys(self, mri_data):
        """Run kidney detection on MRI data"""
        # Normalize
        mri_normalized = (mri_data - mri_data.mean()) / (mri_data.std() + 1e-8)
        
        # Resize to model input size if needed
        target_size = (128, 128, 64)
        if mri_normalized.shape != target_size:
            zoom_factors = [t/s for t, s in zip(target_size, mri_normalized.shape)]
            mri_normalized = zoom(mri_normalized, zoom_factors, order=1)
        
        # Add batch and channel dimensions
        input_tensor = torch.FloatTensor(mri_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Threshold and resize back
        kidney_mask = (prediction > 0.5).astype(np.uint8)
        if kidney_mask.shape != mri_data.shape:
            zoom_factors = [s/t for s, t in zip(mri_data.shape, kidney_mask.shape)]
            kidney_mask = zoom(kidney_mask, zoom_factors, order=0)
        
        confidence = float(prediction.max())
        return kidney_mask, confidence
    
    def _count_kidneys(self, kidney_mask):
        """Count number of kidney regions"""
        if kidney_mask.sum() == 0:
            return 0
        
        labeled_mask, num_regions = label(kidney_mask)
        
        # Filter small regions
        min_size = 100
        valid_regions = 0
        for i in range(1, num_regions + 1):
            if (labeled_mask == i).sum() >= min_size:
                valid_regions += 1
        
        return valid_regions
    
    def _save_ai_results(self, output_file, kidney_mask, num_kidneys, confidence):
        """Save AI results temporarily"""
        results = {
            'ai_kidney_mask': kidney_mask,
            'ai_num_kidneys_detected': num_kidneys,
            'ai_detection_confidence': confidence,
            'ai_detection_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ai_training_f1_score': 0.836  # From training
        }
        
        sio.savemat(output_file, results, format='5')

def main():
    """Main pipeline execution"""
    
    # Configuration
    MODEL_PATH = r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\kidney_unet_model_best.pth"
    
    # Check if file is provided as argument
    if len(sys.argv) > 1:
        input_files = [sys.argv[1]]
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]
        else:
            output_dir = None
    else:
        # Default test files
        test_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
        input_files = [
            os.path.join(test_dir, "withoutROIwithMRI.mat"),
            os.path.join(test_dir, "withROIwithMRI.mat")
        ]
        output_dir = None
    
    # Initialize pipeline
    pipeline = KidneySlavesPipeline(MODEL_PATH)
    
    try:
        results = []
        
        for input_file in input_files:
            if os.path.exists(input_file):
                print(f"\n{'='*60}")
                print(f"🔄 PROCESSING: {os.path.basename(input_file)}")
                print(f"{'='*60}")
                
                output_file = pipeline.process_file(input_file, output_dir)
                results.append(output_file)
            else:
                print(f"❌ File not found: {input_file}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📋 KIDNEY SLAVES PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Processed {len(results)} files successfully")
        print(f"🎯 Kidney slaves created in all files")
        print(f"👁️  Open files in ArbuzGUI to see kidney slaves!")
        
        for i, result_file in enumerate(results, 1):
            print(f"{i}. {result_file}")
        
        print(f"\n🎉 Ready for ArbuzGUI - kidneys appear as slaves!")
        print(f"🧹 Only final .mat files remain (all temporary files cleaned)")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
