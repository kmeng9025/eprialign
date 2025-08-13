#!/usr/bin/env python3
"""
Static Square Mask Pipeline - Demo Implementation
=================================================

This pipeline creates a simple static square mask on MRI data to demonstrate
that the basic workflow is functioning correctly.

Steps:
1. Load MRI data from .mat file
2. Create a static square mask in the center
3. Add the mask as a slave to the Arbuz project
4. Save the result

This serves as a proof-of-concept before implementing more complex detection.

Author: AI Assistant
Date: 2025-08-13
"""

import numpy as np
import scipy.io as sio
import os
import sys
from datetime import datetime
import subprocess

class StaticSquarePipeline:
    """Simple pipeline that adds a static square mask to MRI data"""
    
    def __init__(self):
        print("🔲 Initializing Static Square Pipeline...")
        print("   ✅ Ready to create static square masks")
    
    def create_square_mask(self, shape, square_size=50):
        """
        Create a static square mask in the center of the volume
        
        Args:
            shape: (height, width, depth) of the volume
            square_size: Size of the square in pixels
            
        Returns:
            Binary mask with square in center
        """
        h, w, d = shape
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Calculate center position
        center_h = h // 2
        center_w = w // 2
        center_d = d // 2
        
        # Define square boundaries
        half_size = square_size // 2
        
        h_start = max(0, center_h - half_size)
        h_end = min(h, center_h + half_size)
        w_start = max(0, center_w - half_size)
        w_end = min(w, center_w + half_size)
        
        # Create square across all depth slices (or just middle slices)
        d_start = max(0, center_d - 2)  # 4-slice thick square
        d_end = min(d, center_d + 2)
        
        # Fill the square region
        mask[h_start:h_end, w_start:w_end, d_start:d_end] = 1
        
        coverage = np.sum(mask) / np.prod(shape) * 100
        
        print(f"   🔲 Created square mask: {square_size}x{square_size} pixels")
        print(f"   📊 Coverage: {coverage:.2f}% of volume")
        print(f"   📍 Position: center ({center_h}, {center_w}, {center_d})")
        
        return mask
    
    def process_file(self, input_file, output_dir=None):
        """
        Process a single .mat file and add static square mask
        
        Args:
            input_file: Path to input .mat file
            output_dir: Output directory (optional)
            
        Returns:
            Path to output file with square mask
        """
        print(f"\n🔲 PROCESSING: {os.path.basename(input_file)}")
        print("="*60)
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"static_square_{timestamp}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load the .mat file (using proper structure handling)
            print("📂 Loading MRI data...")
            data = sio.loadmat(input_file, struct_as_record=False, squeeze_me=True)
            
            if 'images' not in data:
                raise ValueError("No 'images' field found in .mat file")
            
            images = data['images']
            if not hasattr(images, '__len__') or len(images) == 0:
                raise ValueError("No images found in file")
            
            # Process the first image
            image_data = images[0]
            if not hasattr(image_data, 'data') or image_data.data is None:
                raise ValueError("No 'data' field found in image")
            
            mri_data = image_data.data
            print(f"   ✅ MRI data loaded: {mri_data.shape}")
            
            # Create static square mask
            print("🔲 Creating static square mask...")
            square_mask = self.create_square_mask(mri_data.shape, square_size=60)
            
            # Create slave structure (as object with attributes, not dict)
            print("🔧 Creating slave structure...")
            
            # Create a structured array for the slave
            slave_dtype = [
                ('Name', 'O'),
                ('ImageType', 'O'), 
                ('Visible', 'O'),
                ('isLoaded', 'O'),
                ('isStore', 'O'),
                ('Selected', 'O'),
                ('data', 'O')
            ]
            
            slave = np.array([(
                'StaticSquare',
                '3DMASK',
                0,
                0, 
                1,
                0,
                square_mask
            )], dtype=slave_dtype)[0]
            
            # Add slave to image
            if not hasattr(image_data, 'slaves') or image_data.slaves is None or (isinstance(image_data.slaves, np.ndarray) and image_data.slaves.size == 0):
                # Create new slaves array
                image_data.slaves = np.array([slave], dtype=object)
                print("   ✅ Created new slaves array with static square")
            else:
                # Append to existing slaves
                existing_slaves = image_data.slaves
                if isinstance(existing_slaves, np.ndarray):
                    new_slaves = np.append(existing_slaves, slave)
                else:
                    new_slaves = np.array([existing_slaves, slave], dtype=object)
                image_data.slaves = new_slaves
                print("   ✅ Added static square to existing slaves")
            
            # Update images array
            images[0] = image_data
            data['images'] = images
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_WITH_STATIC_SQUARE.mat")
            
            # Save the result
            print("💾 Saving result...")
            sio.savemat(output_file, data, format='5')
            
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"   ✅ Saved: {output_file}")
            print(f"   📊 Size: {file_size:.1f} MB")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Error processing {input_file}: {str(e)}")
            raise

def main():
    """Main pipeline execution"""
    
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
    pipeline = StaticSquarePipeline()
    
    try:
        print("🔲 STATIC SQUARE MASK PIPELINE")
        print("="*60)
        print(f"📂 Processing {len(input_files)} file(s)")
        
        results = []
        
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"❌ File not found: {input_file}")
                continue
            
            result = pipeline.process_file(input_file, output_dir)
            results.append(result)
        
        print(f"\n✅ SUCCESS! Static square masks added:")
        print("="*60)
        
        for i, result_file in enumerate(results, 1):
            print(f"{i}. {result_file}")
        
        print(f"\n🎉 Ready for ArbuzGUI - static squares appear as slaves!")
        print(f"👁️  Open files in ArbuzGUI to see the static square masks")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
