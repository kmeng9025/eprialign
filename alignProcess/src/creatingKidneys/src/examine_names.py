"""
Updated kidney mask extraction with proper name checking
"""
import numpy as np
import scipy.io
import os
from pathlib import Path

def examine_slave_names_and_data(filepath):
    """Examine slave names and data to identify true kidney masks"""
    print(f"\n{'='*60}")
    print(f"EXAMINING SLAVE NAMES: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
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
                            print(f"    Found {len(img.slaves)} slaves:")
                            
                            for j, slave in enumerate(img.slaves):
                                if slave is None or (isinstance(slave, np.ndarray) and slave.size == 0):
                                    print(f"      Slave {j}: Empty")
                                    continue
                                
                                # Get slave attributes
                                slave_name = getattr(slave, 'name', 'NO_NAME')
                                slave_type = getattr(slave, 'type', 'NO_TYPE')
                                
                                print(f"      Slave {j}: name='{slave_name}', type='{slave_type}'")
                                
                                # Check if it has data matching MRI dimensions
                                if (hasattr(slave, 'data') and slave.data is not None and
                                    hasattr(slave.data, 'shape')):
                                    
                                    slave_shape = slave.data.shape
                                    if slave_shape == img.data.shape:
                                        unique_vals = np.unique(slave.data)
                                        non_zero_count = np.count_nonzero(slave.data)
                                        total_pixels = slave.data.size
                                        percent_filled = (non_zero_count / total_pixels) * 100
                                        
                                        print(f"        Data: shape={slave_shape}, unique_vals={unique_vals}, coverage={percent_filled:.2f}%")
                                        
                                        # Check if it's a kidney mask (not KidneySRF)
                                        is_kidney_mask = False
                                        exclude_reasons = []
                                        
                                        # Exclude KidneySRF (uppercase)
                                        if 'KidneySRF' in slave_name:
                                            exclude_reasons.append("KidneySRF (not a mask)")
                                        
                                        # Look for kidney-related names (case-insensitive, but not SRF)
                                        elif 'kidney' in slave_name.lower():
                                            if 'srf' not in slave_name.lower():
                                                is_kidney_mask = True
                                            else:
                                                exclude_reasons.append("Contains 'srf' (surface, not mask)")
                                        
                                        # Check if it's binary and reasonable size
                                        elif len(unique_vals) <= 5 and 0.5 <= percent_filled <= 25:
                                            # Could be a kidney mask even without explicit name
                                            is_kidney_mask = True
                                            print(f"        *** POTENTIAL UNNAMED KIDNEY MASK ***")
                                        
                                        if exclude_reasons:
                                            print(f"        EXCLUDED: {', '.join(exclude_reasons)}")
                                        elif is_kidney_mask:
                                            print(f"        *** KIDNEY MASK IDENTIFIED ***")
                                        else:
                                            print(f"        Not a kidney mask (coverage={percent_filled:.2f}%)")
                                    else:
                                        print(f"        Data shape {slave_shape} doesn't match MRI")
                                else:
                                    print(f"        No matching data")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Examine all training files to understand slave naming"""
    training_dir = Path("../../../data/training")
    skip_files = {"withoutROI.mat", "withoutROIwithMRI.mat"}
    
    print("EXAMINING SLAVE NAMES AND DATA")
    print("="*60)
    
    for mat_file in training_dir.glob("*.mat"):
        if mat_file.name not in skip_files:
            examine_slave_names_and_data(mat_file)

if __name__ == "__main__":
    main()
