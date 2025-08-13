"""
Detailed examination of slave data to identify potential kidney masks
"""
import numpy as np
import scipy.io
import os
from pathlib import Path
import matplotlib.pyplot as plt

def examine_slave_data(filepath):
    """Examine slave data in detail to identify kidney masks"""
    print(f"\n{'='*60}")
    print(f"EXAMINING SLAVES: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        
        if 'images' in data:
            images = data['images']
            
            for i in range(len(images)):
                img = images[i]
                
                if img is None or (isinstance(img, np.ndarray) and img.size == 0):
                    continue
                
                # Only examine images with MRI-like data (high resolution)
                if hasattr(img, 'data') and img.data is not None and hasattr(img.data, 'shape'):
                    if len(img.data.shape) == 3 and img.data.shape[0] >= 64:  # Focus on larger images
                        print(f"\nImage {i}: MRI data shape {img.data.shape}")
                        
                        # Check slaves
                        if hasattr(img, 'slaves') and img.slaves is not None:
                            if isinstance(img.slaves, np.ndarray) and img.slaves.size > 0:
                                print(f"  Has {len(img.slaves)} slaves")
                                
                                for j, slave in enumerate(img.slaves):
                                    if slave is None or (isinstance(slave, np.ndarray) and slave.size == 0):
                                        continue
                                    
                                    print(f"    Slave {j}:")
                                    
                                    if hasattr(slave, 'data') and slave.data is not None:
                                        if hasattr(slave.data, 'shape'):
                                            slave_shape = slave.data.shape
                                            print(f"      Data shape: {slave_shape}")
                                            
                                            # Check if slave data matches MRI dimensions
                                            if len(slave_shape) == 3 and slave_shape == img.data.shape:
                                                print(f"      *** MATCHES MRI DIMENSIONS ***")
                                                
                                                # Check data characteristics
                                                try:
                                                    unique_vals = np.unique(slave.data)
                                                    print(f"      Unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
                                                    
                                                    # Check if it's binary-like (good for masks)
                                                    if len(unique_vals) <= 5:
                                                        print(f"      *** BINARY/CATEGORICAL DATA - POTENTIAL MASK ***")
                                                        
                                                        # Count non-zero pixels
                                                        non_zero_count = np.count_nonzero(slave.data)
                                                        total_pixels = slave.data.size
                                                        percent_filled = (non_zero_count / total_pixels) * 100
                                                        
                                                        print(f"      Non-zero pixels: {non_zero_count}/{total_pixels} ({percent_filled:.1f}%)")
                                                        
                                                        # Check if it's a reasonable size for kidney masks (not too small, not too large)
                                                        if 0.1 <= percent_filled <= 30:  # Reasonable kidney mask size
                                                            print(f"      *** REASONABLE MASK SIZE FOR KIDNEYS ***")
                                                            
                                                            # Save this as a potential training sample
                                                            filename = os.path.basename(filepath).replace('.mat', '')
                                                            print(f"      >>> POTENTIAL TRAINING SAMPLE: {filename}_img{i}_slave{j}")
                                                            
                                                            # Check if there are multiple reasonable masks (left/right kidney)
                                                            reasonable_masks = 0
                                                            for k, other_slave in enumerate(img.slaves):
                                                                if (other_slave is not None and 
                                                                    hasattr(other_slave, 'data') and 
                                                                    other_slave.data is not None and
                                                                    hasattr(other_slave.data, 'shape')):
                                                                    
                                                                    if other_slave.data.shape == img.data.shape:
                                                                        other_unique = np.unique(other_slave.data)
                                                                        if len(other_unique) <= 5:
                                                                            other_non_zero = np.count_nonzero(other_slave.data)
                                                                            other_percent = (other_non_zero / total_pixels) * 100
                                                                            if 0.1 <= other_percent <= 30:
                                                                                reasonable_masks += 1
                                                            
                                                            if reasonable_masks >= 2:
                                                                print(f"      >>> MULTIPLE REASONABLE MASKS ({reasonable_masks}) - LIKELY LEFT/RIGHT KIDNEYS!")
                                                    
                                                    else:
                                                        print(f"      Too many unique values ({len(unique_vals)}) - likely not a binary mask")
                                                        
                                                except Exception as e:
                                                    print(f"      Error analyzing data: {e}")
                                            
                                            elif len(slave_shape) == 2:
                                                print(f"      2D data - might be contour or slice")
                                            else:
                                                print(f"      Different dimensions from MRI")
                                        else:
                                            print(f"      Data present but no shape info")
                                    else:
                                        print(f"      No data")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    training_dir = Path("../../../data/training")
    
    # Focus on files that likely have annotations based on our previous exploration
    files_with_slaves = [
        "ExchangeB6M005.mat",
        "HemoB6M022_better.mat", 
        "HemoB6M024.mat",
        "HemoM002.mat",
        "HemoM003.mat",
        "HemoM004.mat"
    ]
    
    print("SEARCHING FOR KIDNEY MASK ANNOTATIONS")
    print("="*60)
    
    for filename in files_with_slaves:
        filepath = training_dir / filename
        if filepath.exists():
            examine_slave_data(filepath)

if __name__ == "__main__":
    main()
