#!/usr/bin/env python3
"""
AI Kidney Detection Pipeline - Final Version
===========================================

This pipeline uses the trained U-Net model and MATLAB integration
to create kidney slaves in Arbuz projects.

Author: AI Assistant
Date: 2025-08-13
"""

import torch
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom, label, binary_dilation
import os
import sys
import subprocess
from datetime import datetime

# Import model architecture
from unet_3d import UNet3D

class AIKidneyDetector:
    """AI kidney detection with MATLAB integration"""
    
    def __init__(self, model_path="kidney_unet_model_best.pth"):
        print("🤖 Initializing AI Kidney Detection...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   🔧 Device: {self.device}")
        
        # Load trained model
        self.model = self._load_model(model_path)
        print("   ✅ AI model loaded successfully")
    
    def _load_model(self, model_path):
        """Load the trained kidney detection model (Random Forest or U-Net)"""
        print(f"   📂 Loading model: {model_path}")
        
        # Check if Random Forest model exists (temporarily disabled for testing)
        rf_model_path = os.path.join(os.path.dirname(model_path), 'kidney_random_forest_best.joblib')
        if False and os.path.exists(rf_model_path):  # Disabled Random Forest for now
            print(f"   🌳 Using Random Forest model: {rf_model_path}")
            from random_forest_kidney import load_random_forest_model
            model = load_random_forest_model(rf_model_path)
            self.model_type = 'random_forest'
            return model
        
        # Fallback to U-Net
        print(f"   🧠 Using U-Net model: {model_path}")
        
        # Initialize model architecture
        model = UNet3D(in_channels=1, out_channels=1)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_loss = checkpoint.get('best_val_loss', 'N/A')
            print(f"   📈 Best validation loss: {best_loss}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        self.model_type = 'unet'
        
        return model
    
    def predict_kidneys(self, mri_data):
        """Run AI prediction on MRI data"""
        print(f"   🧠 Running AI prediction on {mri_data.shape} volume...")
        
        # Handle Random Forest vs U-Net prediction
        if hasattr(self, 'model_type') and self.model_type == 'random_forest':
            # Random Forest prediction
            # Normalize data
            mri_normalized = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
            
            # Use Random Forest prediction
            # Convert to tensor format for compatibility
            input_tensor = torch.FloatTensor(mri_normalized).unsqueeze(0).unsqueeze(0)
            output = self.model(input_tensor)
            
            # Random Forest returns numpy array, not tensor
            if isinstance(output, np.ndarray):
                prediction = output[0, 0]
            else:
                prediction = output.cpu().numpy()[0, 0]
            
            # Threshold for final mask
            threshold = 0.3  # Lower threshold to be more sensitive
            kidney_mask = prediction > threshold
            
        else:
            # U-Net prediction (original code)
            # Normalize data
            mri_normalized = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
            
            # Resize to model target size (64, 64, 32)
            target_size = (64, 64, 32)
            zoom_factors = [t/s for t, s in zip(target_size, mri_data.shape)]
            mri_resized = zoom(mri_normalized, zoom_factors, order=1)
            
            # Prepare for model
            input_tensor = torch.FloatTensor(mri_resized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Run prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            # Resize back to original size
            original_zoom_factors = [s/t for s, t in zip(mri_data.shape, target_size)]
            kidney_prediction = zoom(prediction, original_zoom_factors, order=1)
            
            # Threshold and post-process
            threshold = 0.7  # Increased threshold to handle overconfident model
            kidney_mask = kidney_prediction > threshold
            
            # Clean up prediction with morphological operations
            kidney_mask = binary_dilation(kidney_mask, iterations=1)
        
        # Find connected components (kidneys)
        labeled_mask, num_kidneys = label(kidney_mask)
        
        # Filter small components
        min_kidney_size = 50
        final_mask = np.zeros_like(kidney_mask, dtype=bool)
        valid_kidneys = 0
        
        for i in range(1, num_kidneys + 1):
            component = labeled_mask == i
            if np.sum(component) >= min_kidney_size:
                final_mask |= component
                valid_kidneys += 1
        
        confidence = np.mean(kidney_prediction[final_mask]) if np.any(final_mask) else 0.0
        coverage = np.sum(final_mask) / np.prod(mri_data.shape) * 100
        
        print(f"   🎯 Detected {valid_kidneys} kidneys (confidence: {confidence:.3f})")
        print(f"   📊 Coverage: {coverage:.2f}% of volume")
        
        return final_mask.astype(np.uint8), valid_kidneys, confidence
    
    def process_file(self, input_file, output_dir=None):
        """Process a single .mat file with AI kidney detection"""
        print(f"\n🤖 AI PROCESSING: {os.path.basename(input_file)}")
        print("="*60)
        
        # Set up output directory
        if output_dir is None:
            inference_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(inference_dir, f"ai_kidneys_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"   📁 Output directory: {output_dir}")
        
        try:
            # Load the .mat file
            print("📂 Loading MRI data...")
            data = sio.loadmat(input_file, struct_as_record=False, squeeze_me=True)
            
            if 'images' not in data:
                raise ValueError("No 'images' field found in .mat file")
            
            images = data['images']
            if not hasattr(images, '__len__') or len(images) == 0:
                raise ValueError("No images found in file")
            
            # Find the MRI data
            mri_data = None
            
            for i in range(len(images)):
                img = images[i]
                if hasattr(img, 'data') and img.data is not None:
                    if hasattr(img.data, 'shape') and len(img.data.shape) == 3:
                        mri_data = img.data
                        break
            
            if mri_data is None:
                raise ValueError("No 3D MRI data found")
            
            print(f"   ✅ Found MRI data: {mri_data.shape}")
            
            # Run AI prediction
            kidney_mask, num_kidneys, confidence = self.predict_kidneys(mri_data)
            
            # Save AI results to temporary file
            ai_results_file = os.path.join(output_dir, "ai_kidney_results.mat")
            
            ai_results = {
                'ai_kidney_mask': kidney_mask,
                'ai_num_kidneys_detected': num_kidneys,
                'ai_detection_confidence': confidence,
                'ai_detection_timestamp': datetime.now().isoformat(),
                'ai_training_f1_score': 0.839,
                'ai_coverage_percent': np.sum(kidney_mask) / np.prod(mri_data.shape) * 100,
                'ai_model_info': 'UNet3D trained on kidney MRI data'
            }
            
            print("💾 Saving AI results...")
            sio.savemat(ai_results_file, ai_results, format='5')
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_WITH_AI_KIDNEYS.mat")
            
            # Call MATLAB to create kidney slaves
            print("🔧 Calling MATLAB to create kidney slaves...")
            matlab_cmd = [
                'matlab', '-batch',
                f"addpath('C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\src\\creatingKidneys\\src'); "
                f"create_kidney_slaves_final('{input_file}', '{ai_results_file}', '{output_file}'); "
                f"exit;"
            ]
            
            result = subprocess.run(matlab_cmd, capture_output=True, text=True, 
                                  cwd=os.path.dirname(__file__), encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                print(f"   ✅ MATLAB execution successful")
                # Print relevant output
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if any(marker in line for marker in ['✅', '🎯', '📁', '📊', '🤖', 'kidney', 'slave']):
                            print(f"      {line}")
            else:
                print(f"   ❌ MATLAB error: {result.stderr}")
                raise RuntimeError(f"MATLAB execution failed: {result.stderr}")
            
            # Clean up temporary AI results file
            if os.path.exists(ai_results_file):
                os.remove(ai_results_file)
                print("   🧹 Cleaned up temporary AI results file")
            
            # Verify output file exists
            if os.path.exists(output_file):
                file_size_mb = os.path.getsize(output_file) / (1024*1024)
                print(f"\n✅ SUCCESS! AI kidney detection complete:")
                print(f"   📁 File: {output_file}")
                print(f"   📊 Size: {file_size_mb:.1f} MB")
                print(f"   🤖 AI kidneys: {num_kidneys} detected")
                print(f"   🎯 Confidence: {confidence:.3f}")
                print(f"   👁️  Kidneys will be visible as slaves in ArbuzGUI!")
                
                return output_file
            else:
                raise FileNotFoundError("Output file was not created by MATLAB")
                
        except Exception as e:
            print(f"❌ Error processing {input_file}: {str(e)}")
            raise

def main():
    """Main execution"""
    
    # Check if file is provided as argument
    if len(sys.argv) > 1:
        input_files = [sys.argv[1]]
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]
        else:
            output_dir = None
    else:
        # Default test file
        test_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
        input_files = [os.path.join(test_dir, "withoutROIwithMRI.mat")]
        output_dir = None
    
    # Initialize pipeline
    detector = AIKidneyDetector()
    
    try:
        print("🤖 AI KIDNEY DETECTION PIPELINE")
        print("="*60)
        print(f"📂 Processing {len(input_files)} file(s)")
        
        results = []
        
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"❌ File not found: {input_file}")
                continue
            
            result = detector.process_file(input_file, output_dir)
            results.append(result)
        
        print(f"\n✅ SUCCESS! AI kidney detection complete:")
        print("="*60)
        
        for i, result_file in enumerate(results, 1):
            print(f"{i}. {result_file}")
        
        print(f"\n🎉 Ready for ArbuzGUI - AI kidneys appear as slaves!")
        print(f"👁️  Open files in ArbuzGUI to see AI-detected kidney slaves")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
