"""
Proper kidney mask extraction with correct name identification
"""
import numpy as np
import scipy.io
import os
from pathlib import Path

def extract_true_kidney_masks(filepath):
    """Extract genuine kidney masks by name and type"""
    print(f"\n{'='*60}")
    print(f"EXTRACTING TRUE KIDNEY MASKS: {os.path.basename(filepath)}")
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
                
                # Look for MRI data (350x350)
                if (hasattr(img, 'data') and img.data is not None and 
                    hasattr(img.data, 'shape') and len(img.data.shape) == 3 and 
                    img.data.shape[0] == 350 and img.data.shape[1] == 350):
                    
                    print(f"  MRI Image {i}: shape {img.data.shape}")
                    
                    if hasattr(img, 'slaves') and img.slaves is not None:
                        if isinstance(img.slaves, np.ndarray) and img.slaves.size > 0:
                            print(f"    Checking {len(img.slaves)} slaves...")
                            
                            for j, slave in enumerate(img.slaves):
                                if slave is None or (isinstance(slave, np.ndarray) and slave.size == 0):
                                    continue
                                
                                # Get slave name and type properly
                                slave_name = getattr(slave, 'Name', 'UNKNOWN')
                                slave_type = getattr(slave, 'ImageType', 'UNKNOWN')
                                
                                print(f"      Slave {j}: Name='{slave_name}', Type='{slave_type}'")
                                
                                # Check if this is a kidney mask (not surface)
                                is_kidney_mask = False
                                
                                # Look for "Kidney" name specifically (case-sensitive)
                                if slave_name == 'Kidney' and slave_type == '3DMASK':
                                    is_kidney_mask = True
                                    print(f"        *** CONFIRMED KIDNEY MASK by name and type ***")
                                
                                elif 'Kidney' in slave_name and 'SRF' not in slave_name and slave_type == '3DMASK':
                                    is_kidney_mask = True
                                    print(f"        *** KIDNEY MASK variant found ***")
                                
                                elif 'KidneySRF' in slave_name or slave_type == '3DSURFACE':
                                    print(f"        EXCLUDED: Surface data (not mask)")
                                    continue
                                
                                # Check the data if it's identified as kidney mask
                                if is_kidney_mask and hasattr(slave, 'data') and slave.data is not None:
                                    if hasattr(slave.data, 'shape') and slave.data.shape == img.data.shape:
                                        unique_vals = np.unique(slave.data)
                                        non_zero_count = np.count_nonzero(slave.data)
                                        total_pixels = slave.data.size
                                        percent_filled = (non_zero_count / total_pixels) * 100
                                        
                                        print(f"        Mask data: unique_vals={unique_vals}, coverage={percent_filled:.2f}%")
                                        
                                        # Verify it's a reasonable kidney mask
                                        if len(unique_vals) <= 5 and percent_filled > 0.1:
                                            sample = {
                                                'file': os.path.basename(filepath),
                                                'image_idx': i,
                                                'slave_idx': j,
                                                'mri_data': img.data.copy(),
                                                'kidney_mask': slave.data.copy(),
                                                'mask_name': slave_name,
                                                'mask_type': slave_type,
                                                'mri_shape': img.data.shape,
                                                'mask_coverage': percent_filled
                                            }
                                            
                                            training_samples.append(sample)
                                            print(f"        >>> ADDED TO TRAINING SET <<<")
                                        else:
                                            print(f"        Invalid mask data: {len(unique_vals)} unique values, {percent_filled:.2f}% coverage")
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
    
    return training_samples

def extract_all_kidney_training_data():
    """Extract all kidney masks from training data"""
    training_dir = Path("../../../data/training")
    skip_files = {"withoutROI.mat", "withoutROIwithMRI.mat"}
    
    all_training_samples = []
    
    print("TRUE KIDNEY MASK EXTRACTION")
    print("="*60)
    print("Looking for named 'Kidney' masks in MRI images...")
    
    for mat_file in training_dir.glob("*.mat"):
        if mat_file.name not in skip_files:
            samples = extract_true_kidney_masks(mat_file)
            all_training_samples.extend(samples)
    
    print(f"\n{'='*60}")
    print(f"FINAL EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total kidney mask training samples: {len(all_training_samples)}")
    
    for i, sample in enumerate(all_training_samples):
        print(f"  Sample {i+1}: {sample['file']} - MRI {sample['image_idx']}")
        print(f"    Mask: '{sample['mask_name']}' ({sample['mask_type']}) - {sample['mask_coverage']:.1f}% coverage")
        print(f"    Shape: {sample['mri_shape']}")
    
    return all_training_samples

def main():
    """Main extraction function"""
    samples = extract_all_kidney_training_data()
    
    if len(samples) > 0:
        print(f"\n🎉 SUCCESS: Found {len(samples)} kidney mask training samples!")
        print("Ready to train the model...")
        
        # Save the samples for later use
        import pickle
        output_file = "kidney_training_samples.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Training samples saved to: {output_file}")
        
    else:
        print("\n❌ No kidney mask training samples found!")
    
    return samples

if __name__ == "__main__":
    main()
