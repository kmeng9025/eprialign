"""
Better approach to extract slave names and identify kidney masks
"""
import numpy as np
import scipy.io
import os
from pathlib import Path

def examine_slave_structure(filepath):
    """Examine the actual structure of slaves to get proper names"""
    print(f"\n{'='*60}")
    print(f"DETAILED SLAVE EXAMINATION: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        # Try different loading methods
        data1 = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        data2 = scipy.io.loadmat(filepath, struct_as_record=True, squeeze_me=False)
        
        if 'images' in data1:
            images = data1['images']
            
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
                        print(f"    Slaves type: {type(img.slaves)}")
                        print(f"    Slaves shape: {getattr(img.slaves, 'shape', 'no shape')}")
                        
                        if isinstance(img.slaves, np.ndarray) and img.slaves.size > 0:
                            print(f"    Found {len(img.slaves)} slaves:")
                            
                            for j, slave in enumerate(img.slaves):
                                if slave is None or (isinstance(slave, np.ndarray) and slave.size == 0):
                                    print(f"      Slave {j}: Empty")
                                    continue
                                
                                print(f"      Slave {j}: type={type(slave)}")
                                
                                # Try different ways to access name/type
                                slave_name = "UNKNOWN"
                                slave_type = "UNKNOWN"
                                
                                # Method 1: Direct attribute access
                                if hasattr(slave, 'name'):
                                    slave_name = slave.name
                                    print(f"        Direct name: '{slave_name}'")
                                
                                if hasattr(slave, 'type'):
                                    slave_type = slave.type
                                    print(f"        Direct type: '{slave_type}'")
                                
                                # Method 2: Check if it's a matlab struct
                                if hasattr(slave, '_fieldnames'):
                                    print(f"        Fieldnames: {slave._fieldnames}")
                                    for field in slave._fieldnames:
                                        try:
                                            val = getattr(slave, field)
                                            print(f"        {field}: {val}")
                                        except:
                                            print(f"        {field}: (access error)")
                                
                                # Method 3: Check if it's numpy structured array
                                if hasattr(slave, 'dtype') and hasattr(slave.dtype, 'names') and slave.dtype.names:
                                    print(f"        Dtype names: {slave.dtype.names}")
                                    for field in slave.dtype.names:
                                        try:
                                            val = slave[field][0] if hasattr(slave[field], '__len__') else slave[field]
                                            print(f"        {field}: {val}")
                                        except:
                                            print(f"        {field}: (access error)")
                                
                                # Check data regardless of name
                                if (hasattr(slave, 'data') and slave.data is not None and
                                    hasattr(slave.data, 'shape')):
                                    
                                    slave_shape = slave.data.shape
                                    if slave_shape == img.data.shape:
                                        unique_vals = np.unique(slave.data)
                                        non_zero_count = np.count_nonzero(slave.data)
                                        total_pixels = slave.data.size
                                        percent_filled = (non_zero_count / total_pixels) * 100
                                        
                                        print(f"        Data: shape={slave_shape}, unique={unique_vals}, coverage={percent_filled:.2f}%")
                                        
                                        # Check if it could be a kidney mask
                                        if len(unique_vals) <= 5 and 0.5 <= percent_filled <= 25:
                                            print(f"        *** POTENTIAL KIDNEY MASK ***")
                                            
                                            # Check name for kidney/KidneySRF
                                            name_str = str(slave_name).lower()
                                            if 'kidney' in name_str:
                                                if 'srf' in name_str:
                                                    print(f"        EXCLUDED: Contains 'srf' (surface data)")
                                                else:
                                                    print(f"        *** CONFIRMED KIDNEY MASK by name ***")
                                            else:
                                                print(f"        *** UNNAMED KIDNEY MASK (by size/coverage) ***")
                                
                                print()  # separator
        
        # Also try with the second loading method for comparison
        print(f"\n    --- Trying struct_as_record=True ---")
        if 'images' in data2:
            images2 = data2['images']
            print(f"    Images2 type: {type(images2)}")
            print(f"    Images2 shape: {getattr(images2, 'shape', 'no shape')}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Examine one file in detail to understand the structure"""
    training_dir = Path("../../../data/training")
    
    # Pick one file that we know has kidney masks
    test_file = "HemoB6M022_better.mat"
    filepath = training_dir / test_file
    
    if filepath.exists():
        examine_slave_structure(filepath)
    else:
        print(f"File not found: {test_file}")

if __name__ == "__main__":
    main()
