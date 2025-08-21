#!/usr/bin/env python3
"""
Extract kidney training data from .mat files and convert to numpy arrays
"""

import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
import pickle

def load_mat_file(filepath):
    """Load .mat file and return the data structure"""
    try:
        return sio.loadmat(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def find_kidney_slaves(mat_data):
    """Find kidney slaves in the loaded mat data"""
    kidney_data = []
    
    # Look for main data structure
    main_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if not main_keys:
        return kidney_data
    
    main_data = mat_data[main_keys[0]]
    
    # Navigate the structure to find slaves
    if hasattr(main_data, 'dtype') and main_data.dtype.names:
        # Look for slaves field
        if 'slaves' in main_data.dtype.names:
            slaves = main_data['slaves'][0, 0]
            if slaves.size > 0:
                for i in range(slaves.shape[0]):
                    slave = slaves[i, 0]
                    if hasattr(slave, 'dtype') and slave.dtype.names:
                        # Check if this slave has kidney-related name
                        if 'name' in slave.dtype.names:
                            name = slave['name'][0, 0]
                            if isinstance(name, np.ndarray) and name.size > 0:
                                name_str = str(name[0]) if name.ndim > 0 else str(name)
                                if 'kidney' in name_str.lower():
                                    # Extract the mask data
                                    if 'mask' in slave.dtype.names:
                                        mask = slave['mask'][0, 0]
                                        if mask.size > 0:
                                            kidney_data.append({
                                                'name': name_str,
                                                'mask': mask,
                                                'slave_index': i
                                            })
                                            print(f"   Found kidney slave: {name_str} with mask shape {mask.shape}")
    
    return kidney_data

def find_mri_images(mat_data):
    """Find MRI images in the loaded mat data"""
    mri_images = []
    
    # Look for main data structure
    main_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if not main_keys:
        return mri_images
    
    main_data = mat_data[main_keys[0]]
    
    # Navigate the structure to find images
    if hasattr(main_data, 'dtype') and main_data.dtype.names:
        if 'images' in main_data.dtype.names:
            images = main_data['images'][0, 0]
            if images.size > 0:
                for i in range(images.shape[0]):
                    image = images[i, 0]
                    if hasattr(image, 'dtype') and image.dtype.names:
                        # Get image name and data
                        if 'name' in image.dtype.names and 'data' in image.dtype.names:
                            name = image['name'][0, 0]
                            data = image['data'][0, 0]
                            
                            if isinstance(name, np.ndarray) and name.size > 0:
                                name_str = str(name[0]) if name.ndim > 0 else str(name)
                                if 'mri' in name_str.lower() and data.size > 0:
                                    mri_images.append({
                                        'name': name_str,
                                        'data': data,
                                        'image_index': i
                                    })
                                    print(f"   Found MRI image: {name_str} with shape {data.shape}")
    
    return mri_images

def normalize_data(data):
    """Normalize data to [0, 1] range"""
    if data.max() == data.min():
        return np.zeros_like(data, dtype=np.float32)
    
    normalized = (data - data.min()) / (data.max() - data.min())
    return normalized.astype(np.float32)

def resize_to_standard(data, target_size=(128, 128, 32)):
    """Resize data to standard size for training"""
    if data.shape == target_size:
        return data
    
    zoom_factors = [t/s for s, t in zip(data.shape, target_size)]
    resized = zoom(data, zoom_factors, order=1)
    return resized

def extract_training_pairs(mat_file_path):
    """Extract MRI-kidney pairs from a single .mat file"""
    print(f"\n🔍 Processing: {os.path.basename(mat_file_path)}")
    
    # Load the mat file
    mat_data = load_mat_file(mat_file_path)
    if mat_data is None:
        return []
    
    # Find MRI images and kidney slaves
    mri_images = find_mri_images(mat_data)
    kidney_slaves = find_kidney_slaves(mat_data)
    
    if not mri_images:
        print("   ❌ No MRI images found")
        return []
    
    if not kidney_slaves:
        print("   ❌ No kidney slaves found")
        return []
    
    training_pairs = []
    
    # Create training pairs for each MRI image
    for mri in mri_images:
        mri_data = mri['data']
        mri_name = mri['name']
        
        print(f"   📊 MRI {mri_name}: shape {mri_data.shape}")
        
        # Create combined kidney mask for this MRI
        combined_mask = np.zeros_like(mri_data, dtype=np.float32)
        kidney_count = 0
        
        for kidney in kidney_slaves:
            kidney_mask = kidney['mask']
            kidney_name = kidney['name']
            
            # Check if kidney mask matches MRI dimensions
            if kidney_mask.shape == mri_data.shape:
                # Add kidney mask to combined mask
                combined_mask = np.maximum(combined_mask, kidney_mask.astype(np.float32))
                kidney_count += 1
                print(f"      ✅ Added kidney {kidney_name} to mask")
            else:
                print(f"      ⚠️  Kidney {kidney_name} shape {kidney_mask.shape} doesn't match MRI shape {mri_data.shape}")
        
        if kidney_count > 0:
            # Normalize and resize
            mri_normalized = normalize_data(mri_data)
            mask_normalized = combined_mask  # Keep as 0/1
            
            # Resize to standard size
            target_size = (128, 128, 32)
            mri_resized = resize_to_standard(mri_normalized, target_size)
            mask_resized = resize_to_standard(mask_normalized, target_size)
            
            # Threshold mask after resizing
            mask_resized = (mask_resized > 0.5).astype(np.float32)
            
            training_pairs.append({
                'mri': mri_resized,
                'mask': mask_resized,
                'source_file': os.path.basename(mat_file_path),
                'mri_name': mri_name,
                'kidney_count': kidney_count,
                'original_shape': mri_data.shape
            })
            
            print(f"   ✅ Created training pair: MRI {mri_name} with {kidney_count} kidneys")
            print(f"      Original: {mri_data.shape} → Resized: {mri_resized.shape}")
            print(f"      Mask coverage: {np.sum(mask_resized) / np.prod(mask_resized.shape) * 100:.2f}%")
    
    return training_pairs

def main():
    """Main extraction function"""
    print("🔍 KIDNEY TRAINING DATA EXTRACTION")
    print("=" * 50)
    
    # Define paths
    training_data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
    output_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\training\kidneyTrainingData"
    
    # Find all .mat files
    mat_files = []
    if os.path.exists(training_data_dir):
        for file in os.listdir(training_data_dir):
            if file.endswith('.mat'):
                mat_files.append(os.path.join(training_data_dir, file))
    
    print(f"📂 Found {len(mat_files)} .mat files in {training_data_dir}")
    
    if not mat_files:
        print("❌ No .mat files found!")
        return
    
    # Extract training data
    all_training_pairs = []
    
    for mat_file in mat_files:
        try:
            pairs = extract_training_pairs(mat_file)
            all_training_pairs.extend(pairs)
        except Exception as e:
            print(f"❌ Error processing {mat_file}: {e}")
    
    print(f"\n📊 EXTRACTION SUMMARY")
    print("=" * 30)
    print(f"Total training pairs extracted: {len(all_training_pairs)}")
    
    if not all_training_pairs:
        print("❌ No valid training pairs extracted!")
        return
    
    # Save training data
    for i, pair in enumerate(all_training_pairs):
        # Save MRI data
        mri_path = os.path.join(output_dir, f"mri_{i:03d}.npy")
        np.save(mri_path, pair['mri'])
        
        # Save mask data
        mask_path = os.path.join(output_dir, f"mask_{i:03d}.npy")
        np.save(mask_path, pair['mask'])
        
        print(f"✅ Saved pair {i:03d}: {pair['source_file']} - {pair['mri_name']}")
        print(f"   MRI: {mri_path}")
        print(f"   Mask: {mask_path}")
        print(f"   Shape: {pair['mri'].shape}, Kidney coverage: {np.sum(pair['mask']) / np.prod(pair['mask'].shape) * 100:.2f}%")
    
    # Save metadata
    metadata = {
        'total_pairs': len(all_training_pairs),
        'pairs': [
            {
                'index': i,
                'source_file': pair['source_file'],
                'mri_name': pair['mri_name'],
                'kidney_count': pair['kidney_count'],
                'original_shape': pair['original_shape'],
                'final_shape': pair['mri'].shape,
                'mask_coverage': float(np.sum(pair['mask']) / np.prod(pair['mask'].shape))
            }
            for i, pair in enumerate(all_training_pairs)
        ]
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Saved metadata: {metadata_path}")
    print("🎉 Training data extraction complete!")
    
    return all_training_pairs

if __name__ == "__main__":
    main()
