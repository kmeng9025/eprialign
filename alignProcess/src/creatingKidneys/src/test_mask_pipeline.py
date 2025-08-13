#!/usr/bin/env python3
"""
Enhanced Test Mask Pipeline - Multiple Mask Types
=================================================

This pipeline creates various test masks to demonstrate different geometries:
1. Static square mask (center)
2. Static circle mask (upper left) 
3. Static line mask (diagonal)
4. Multiple small dots

This helps verify the mask creation workflow with different shapes.

Author: AI Assistant
Date: 2025-08-13
"""

import numpy as np
import scipy.io as sio
import os
import sys
from datetime import datetime

class TestMaskPipeline:
    """Pipeline that creates multiple test masks to verify workflow"""
    
    def __init__(self):
        print("🎯 Initializing Test Mask Pipeline...")
        print("   ✅ Ready to create various test masks")
    
    def create_square_mask(self, shape, size=50, position='center'):
        """Create a square mask"""
        h, w, d = shape
        mask = np.zeros(shape, dtype=np.uint8)
        
        if position == 'center':
            center_h, center_w, center_d = h//2, w//2, d//2
        elif position == 'upper_left':
            center_h, center_w, center_d = h//4, w//4, d//2
        elif position == 'lower_right':
            center_h, center_w, center_d = 3*h//4, 3*w//4, d//2
            
        half_size = size // 2
        h_start = max(0, center_h - half_size)
        h_end = min(h, center_h + half_size)
        w_start = max(0, center_w - half_size)
        w_end = min(w, center_w + half_size)
        d_start = max(0, center_d - 2)
        d_end = min(d, center_d + 2)
        
        mask[h_start:h_end, w_start:w_end, d_start:d_end] = 1
        return mask
    
    def create_circle_mask(self, shape, radius=25, position='upper_left'):
        """Create a circular mask"""
        h, w, d = shape
        mask = np.zeros(shape, dtype=np.uint8)
        
        if position == 'upper_left':
            center_h, center_w, center_d = h//4, w//4, d//2
        elif position == 'upper_right':
            center_h, center_w, center_d = h//4, 3*w//4, d//2
        elif position == 'center':
            center_h, center_w, center_d = h//2, w//2, d//2
            
        # Create circle in a few middle slices
        for dz in range(max(0, center_d-2), min(d, center_d+3)):
            for y in range(h):
                for x in range(w):
                    if (y - center_h)**2 + (x - center_w)**2 <= radius**2:
                        mask[y, x, dz] = 1
        
        return mask
    
    def create_line_mask(self, shape, thickness=3):
        """Create a diagonal line mask"""
        h, w, d = shape
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Create diagonal line across middle slices
        d_start = max(0, d//2 - 2)
        d_end = min(d, d//2 + 3)
        
        for dz in range(d_start, d_end):
            for i in range(min(h, w)):
                for t in range(-thickness//2, thickness//2 + 1):
                    y = i
                    x = i + t
                    if 0 <= y < h and 0 <= x < w:
                        mask[y, x, dz] = 1
        
        return mask
    
    def create_dots_mask(self, shape, dot_size=5, num_dots=4):
        """Create multiple small dot masks"""
        h, w, d = shape
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Create dots at different positions
        positions = [
            (h//4, w//4, d//2),      # Upper left
            (h//4, 3*w//4, d//2),    # Upper right  
            (3*h//4, w//4, d//2),    # Lower left
            (3*h//4, 3*w//4, d//2)   # Lower right
        ]
        
        for center_h, center_w, center_d in positions[:num_dots]:
            h_start = max(0, center_h - dot_size//2)
            h_end = min(h, center_h + dot_size//2)
            w_start = max(0, center_w - dot_size//2) 
            w_end = min(w, center_w + dot_size//2)
            d_start = max(0, center_d - 1)
            d_end = min(d, center_d + 2)
            
            mask[h_start:h_end, w_start:w_end, d_start:d_end] = 1
        
        return mask
    
    def create_slave(self, name, mask):
        """Create a slave structure for the mask"""
        slave_dtype = [
            ('Name', 'O'),
            ('ImageType', 'O'), 
            ('Visible', 'O'),
            ('isLoaded', 'O'),
            ('isStore', 'O'),
            ('Selected', 'O'),
            ('data', 'O')
        ]
        
        return np.array([(
            name,
            '3DMASK',
            0,
            0, 
            1,
            0,
            mask
        )], dtype=slave_dtype)[0]
    
    def process_file(self, input_file, output_dir=None):
        """Process file and add multiple test masks"""
        print(f"\n🎯 PROCESSING: {os.path.basename(input_file)}")
        print("="*60)
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"test_masks_{timestamp}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load the .mat file
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
            
            # Create different test masks
            print("🎯 Creating test masks...")
            
            # 1. Square mask
            square_mask = self.create_square_mask(mri_data.shape, size=40, position='center')
            square_slave = self.create_slave('TestSquare', square_mask)
            coverage = np.sum(square_mask) / np.prod(mri_data.shape) * 100
            print(f"   🔲 Square mask: {coverage:.2f}% coverage")
            
            # 2. Circle mask  
            circle_mask = self.create_circle_mask(mri_data.shape, radius=20, position='upper_left')
            circle_slave = self.create_slave('TestCircle', circle_mask)
            coverage = np.sum(circle_mask) / np.prod(mri_data.shape) * 100
            print(f"   🔵 Circle mask: {coverage:.2f}% coverage")
            
            # 3. Line mask
            line_mask = self.create_line_mask(mri_data.shape, thickness=3)
            line_slave = self.create_slave('TestLine', line_mask)
            coverage = np.sum(line_mask) / np.prod(mri_data.shape) * 100
            print(f"   📏 Line mask: {coverage:.2f}% coverage")
            
            # 4. Dots mask
            dots_mask = self.create_dots_mask(mri_data.shape, dot_size=8, num_dots=4)
            dots_slave = self.create_slave('TestDots', dots_mask)
            coverage = np.sum(dots_mask) / np.prod(mri_data.shape) * 100
            print(f"   🔘 Dots mask: {coverage:.2f}% coverage")
            
            # Add all slaves to image
            new_slaves = [square_slave, circle_slave, line_slave, dots_slave]
            
            if not hasattr(image_data, 'slaves') or image_data.slaves is None or (isinstance(image_data.slaves, np.ndarray) and image_data.slaves.size == 0):
                # Create new slaves array
                image_data.slaves = np.array(new_slaves, dtype=object)
                print("   ✅ Created new slaves array with 4 test masks")
            else:
                # Append to existing slaves
                existing_slaves = list(image_data.slaves)
                existing_slaves.extend(new_slaves)
                image_data.slaves = np.array(existing_slaves, dtype=object)
                print(f"   ✅ Added 4 test masks to existing slaves")
            
            # Update images array
            images[0] = image_data
            data['images'] = images
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_WITH_TEST_MASKS.mat")
            
            # Save the result
            print("💾 Saving result...")
            sio.savemat(output_file, data, format='5')
            
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"   ✅ Saved: {output_file}")
            print(f"   📊 Size: {file_size:.1f} MB")
            print(f"   🎯 Added 4 test masks: Square, Circle, Line, Dots")
            
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
    pipeline = TestMaskPipeline()
    
    try:
        print("🎯 TEST MASK PIPELINE")
        print("="*60)
        print(f"📂 Processing {len(input_files)} file(s)")
        
        results = []
        
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"❌ File not found: {input_file}")
                continue
            
            result = pipeline.process_file(input_file, output_dir)
            results.append(result)
        
        print(f"\n✅ SUCCESS! Test masks added:")
        print("="*60)
        
        for i, result_file in enumerate(results, 1):
            print(f"{i}. {result_file}")
        
        print(f"\n🎉 Ready for ArbuzGUI - test masks appear as slaves!")
        print(f"👁️  Open files in ArbuzGUI to see: Square, Circle, Line, and Dots masks")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
