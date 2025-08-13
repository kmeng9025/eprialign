"""
Extract kidney masks from MRI data for training
Focus on 350x350 MRI images and look for kidney annotations
"""
import numpy as np
import scipy.io
import os
from pathlib import Path

def extract_kidney_masks(filepath):
    """Extract kidney masks from MRI data"""
    print(f"\n{'='*60}")
    print(f"EXTRACTING KIDNEY MASKS: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    training_samples = []
    
    try:
        data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        
        if 'images' in data:
            images = data['images']
            
            for i in range(len(images)):
                img = images[i]
                
                if img is None or (isinstance(img, np.ndarray) and img.size == 0):
                    continue
                
                # Look for MRI data (350x350 resolution)
                if hasattr(img, 'data') and img.data is not None and hasattr(img.data, 'shape'):
                    shape = img.data.shape
                    
                    # Check if this is MRI data (350x350x*)
                    if len(shape) == 3 and shape[0] == 350 and shape[1] == 350:
                        print(f"  Found MRI Image {i}: shape {shape}")
                        
                        # Look for kidney masks in slaves
                        if hasattr(img, 'slaves') and img.slaves is not None:
                            if isinstance(img.slaves, np.ndarray) and img.slaves.size > 0:
                                print(f"    Checking {len(img.slaves)} slaves for kidney masks...")
                                
                                for j, slave in enumerate(img.slaves):
                                    if slave is None or (isinstance(slave, np.ndarray) and slave.size == 0):
                                        continue
                                    
                                    # Check slave name and type
                                    slave_name = getattr(slave, 'name', 'Unnamed')
                                    slave_type = getattr(slave, 'type', 'Unknown')
                                    
                                    print(f"      Slave {j}: '{slave_name}' (type: {slave_type})")
                                    
                                    # Look for kidney-related names (but not kidneySRF)
                                    is_kidney_mask = False
                                    if 'kidney' in slave_name.lower():
                                        if 'srf' not in slave_name.lower():  # Exclude kidneySRF
                                            is_kidney_mask = True
                                            print(f"        *** POTENTIAL KIDNEY MASK ***")
                                    
                                    # Even if name doesn't indicate kidney, check the data
                                    if hasattr(slave, 'data') and slave.data is not None:
                                        if hasattr(slave.data, 'shape'):
                                            slave_shape = slave.data.shape
                                            print(f"        Data shape: {slave_shape}")
                                            
                                            # Check if slave data matches MRI dimensions
                                            if slave_shape == shape:
                                                print(f"        *** MATCHES MRI DIMENSIONS ***")
                                                
                                                try:
                                                    unique_vals = np.unique(slave.data)
                                                    print(f"        Unique values: {unique_vals}")
                                                    
                                                    # Check if it's binary or has few values (mask-like)
                                                    if len(unique_vals) <= 5:
                                                        non_zero_count = np.count_nonzero(slave.data)
                                                        total_pixels = slave.data.size
                                                        percent_filled = (non_zero_count / total_pixels) * 100
                                                        
                                                        print(f"        Non-zero pixels: {non_zero_count}/{total_pixels} ({percent_filled:.2f}%)")
                                                        
                                                        # Reasonable kidney mask size (0.5% to 25% of image)
                                                        if 0.5 <= percent_filled <= 25:
                                                            print(f"        *** REASONABLE MASK SIZE FOR KIDNEY ***")
                                                            
                                                            # This looks like a kidney mask!
                                                            sample = {
                                                                'file': os.path.basename(filepath),
                                                                'image_idx': i,
                                                                'slave_idx': j,
                                                                'mri_data': img.data,
                                                                'kidney_mask': slave.data,
                                                                'mask_name': slave_name,
                                                                'mask_type': slave_type,
                                                                'mri_shape': shape,
                                                                'mask_coverage': percent_filled
                                                            }
                                                            
                                                            training_samples.append(sample)
                                                            print(f"        >>> ADDED TO TRAINING SET <<<")
                                                        else:
                                                            print(f"        Mask size not reasonable for kidney ({percent_filled:.2f}%)")
                                                    else:
                                                        print(f"        Too many unique values ({len(unique_vals)}) - not a binary mask")
                                                
                                                except Exception as e:
                                                    print(f"        Error analyzing mask data: {e}")
                                            else:
                                                print(f"        Different shape from MRI: {slave_shape}")
                            else:
                                print(f"    No slaves found")
                        else:
                            print(f"    No slaves to check")
                    
                    elif len(shape) == 3 and shape[0] == 64:
                        print(f"  Found EPR Image {i}: shape {shape} (skipping)")
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
    
    return training_samples

def main():
    """Extract all kidney masks from training data"""
    training_dir = Path("../../../data/training")
    
    # Skip the files without annotations
    skip_files = {"withoutROI.mat", "withoutROIwithMRI.mat"}
    
    all_training_samples = []
    
    print("KIDNEY MASK EXTRACTION FOR TRAINING")
    print("="*60)
    print("Looking for MRI images (350x350) with kidney mask annotations...")
    
    for mat_file in training_dir.glob("*.mat"):
        if mat_file.name not in skip_files:
            samples = extract_kidney_masks(mat_file)
            all_training_samples.extend(samples)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total training samples found: {len(all_training_samples)}")
    
    for i, sample in enumerate(all_training_samples):
        print(f"  Sample {i+1}: {sample['file']} - MRI {sample['image_idx']}, Mask '{sample['mask_name']}' ({sample['mask_coverage']:.1f}% coverage)")
    
    # Save summary to file for reference
    if all_training_samples:
        print(f"\nFound {len(all_training_samples)} training samples!")
        print("Ready to create training dataset...")
    else:
        print("\nNo training samples found. Check the data structure.")
    
    return all_training_samples

if __name__ == "__main__":
    samples = main()
