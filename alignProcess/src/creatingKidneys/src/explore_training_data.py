"""
Explore training data to understand structure and identify files with MRI and kidney annotations
"""
import numpy as np
import scipy.io
import os
from pathlib import Path

def explore_mat_file(filepath):
    """Explore a single .mat file to understand its structure"""
    print(f"\n{'='*60}")
    print(f"EXPLORING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        
        # Look for project data
        if 'Project' in data:
            project = data['Project']
            print(f"Project type: {getattr(project, 'type', 'Unknown')}")
            
            # Explore images
            if hasattr(project, 'img') and project.img is not None:
                images = project.img
                if not isinstance(images, np.ndarray):
                    images = [images]
                
                print(f"Number of images: {len(images)}")
                
                for i, img in enumerate(images):
                    if img is None:
                        continue
                        
                    img_type = getattr(img, 'type', 'Unknown')
                    print(f"  Image {i}: type = '{img_type}'")
                    
                    # Check for MRI data
                    if hasattr(img, 'data') and img.data is not None:
                        data_keys = []
                        if isinstance(img.data, np.ndarray) and img.data.dtype.names:
                            data_keys = list(img.data.dtype.names)
                        elif hasattr(img.data, '_fieldnames'):
                            data_keys = img.data._fieldnames
                        
                        mri_keys = [k for k in data_keys if 'MRI' in k or 'mri' in k]
                        if mri_keys:
                            print(f"    MRI data found: {mri_keys}")
                            
                            for key in mri_keys:
                                try:
                                    if hasattr(img.data, key):
                                        mri_data = getattr(img.data, key)
                                    else:
                                        mri_data = img.data[key][0] if isinstance(img.data, np.ndarray) else None
                                    
                                    if mri_data is not None and hasattr(mri_data, 'shape'):
                                        print(f"      {key}: shape {mri_data.shape}")
                                except Exception as e:
                                    print(f"      {key}: Error accessing - {e}")
                    
                    # Check for slaves (annotations/masks)
                    if hasattr(img, 'slaves') and img.slaves is not None:
                        slaves = img.slaves
                        if not isinstance(slaves, np.ndarray):
                            slaves = [slaves] if slaves is not None else []
                        
                        print(f"    Slaves found: {len(slaves)}")
                        
                        for j, slave in enumerate(slaves):
                            if slave is None:
                                continue
                                
                            slave_type = getattr(slave, 'type', 'Unknown')
                            slave_name = getattr(slave, 'name', 'Unnamed')
                            print(f"      Slave {j}: '{slave_name}' (type: {slave_type})")
                            
                            # Check if this could be kidney annotation
                            if 'kidney' in slave_name.lower() or 'renal' in slave_name.lower():
                                print(f"        *** POTENTIAL KIDNEY ANNOTATION ***")
                                
                                if hasattr(slave, 'data') and slave.data is not None:
                                    try:
                                        if hasattr(slave.data, 'shape'):
                                            print(f"        Data shape: {slave.data.shape}")
                                        elif isinstance(slave.data, np.ndarray):
                                            print(f"        Data shape: {slave.data.shape}")
                                    except:
                                        print(f"        Data present but shape unknown")
            else:
                print("No images found in project")
                
    except Exception as e:
        print(f"Error loading file: {e}")
    
    return True

def main():
    """Explore all training data files"""
    training_dir = Path("../../../data/training")
    
    print("TRAINING DATA EXPLORATION")
    print("="*60)
    print(f"Training directory: {training_dir.absolute()}")
    
    mat_files = list(training_dir.glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files")
    
    for mat_file in mat_files:
        explore_mat_file(mat_file)
    
    print(f"\n{'='*60}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
