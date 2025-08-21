#!/usr/bin/env python3
"""
Create proper training data from kidney masks found in .mat files
"""

import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
import pickle

def extract_slave_name(slave):
    """Extract slave name from complex structure"""
    try:
        if hasattr(slave, 'dtype') and slave.dtype.names:
            for name_field in ['name', 'Name', 'slave_name', 'slavename']:
                if name_field in slave.dtype.names:
                    name_data = slave[name_field][0, 0]
                    
                    if isinstance(name_data, (str, np.str_)):
                        return str(name_data)
                    elif isinstance(name_data, np.ndarray):
                        if name_data.size > 0:
                            if name_data.dtype.kind in ['U', 'S']:
                                return str(name_data.flat[0])
                            elif name_data.dtype == object:
                                return str(name_data.flat[0])
    except:
        pass
    return "Unknown"

def extract_slave_mask(slave):
    """Extract slave mask from complex structure"""
    try:
        if hasattr(slave, 'dtype') and slave.dtype.names:
            for mask_field in ['mask', 'Mask', 'data', 'Data']:
                if mask_field in slave.dtype.names:
                    mask_data = slave[mask_field][0, 0]
                    
                    if isinstance(mask_data, np.ndarray) and mask_data.size > 0:
                        return mask_data
    except:
        pass
    return None

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

def extract_training_data_from_file(mat_file_path):
    """Extract all valid kidney training pairs from a .mat file"""
    print(f"\n🔍 Processing: {os.path.basename(mat_file_path)}")
    
    try:
        data = sio.loadmat(mat_file_path)
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        return []
    
    training_pairs = []
    
    # Look at images
    if 'images' in data:
        images = data['images']
        
        for i in range(images.shape[1]):
            try:
                img = images[0, i]
                
                # Get image name
                img_name = "Unknown"
                if hasattr(img, 'dtype') and img.dtype.names:
                    for name_field in ['Name', 'name', 'FileName']:
                        if name_field in img.dtype.names:
                            try:
                                name_data = img[name_field][0, 0]
                                if isinstance(name_data, np.ndarray) and name_data.size > 0:
                                    img_name = str(name_data.flat[0])
                                    break
                            except:
                                pass
                
                # Get image data
                img_data = None
                if hasattr(img, 'dtype') and img.dtype.names and 'data' in img.dtype.names:
                    try:
                        img_data = img['data'][0, 0]
                        if not isinstance(img_data, np.ndarray) or img_data.size == 0:
                            continue
                    except:
                        continue
                
                # Skip empty images
                if img_data is None or img_data.shape == (0, 0):
                    continue
                
                # Check if this is a valid MRI image (case insensitive)
                # Must contain "MRI", can contain "Short", but NOT "Long"
                img_name_lower = img_name.lower()
                
                # Check if it's an MRI image
                is_mri = "mri" in img_name_lower
                has_long = "long" in img_name_lower
                
                # Skip if not MRI or has "long" in the name
                if not is_mri or has_long:
                    print(f"   ⏭️  Skipping image {i}: '{img_name}' (not MRI or contains 'long')")
                    continue
                
                print(f"   📷 Processing MRI Image {i}: '{img_name}' - shape: {img_data.shape}")
                
                # Look for kidney slaves
                if hasattr(img, 'dtype') and img.dtype.names and 'slaves' in img.dtype.names:
                    slaves = img['slaves'][0, 0]
                    
                    if hasattr(slaves, 'shape') and slaves.size > 0:
                        print(f"      🔍 Found slaves array with shape: {slaves.shape}")
                        
                        # Check ALL slaves in the array (iterate through all dimensions)
                        if len(slaves.shape) == 2:  # (rows, cols)
                            for row in range(slaves.shape[0]):
                                for col in range(slaves.shape[1]):
                                    try:
                                        slave = slaves[row, col]
                                        if slave.size > 0:  # Only process non-empty slaves
                                            
                                            # Extract slave name and mask
                                            slave_name = extract_slave_name(slave)
                                            slave_mask = extract_slave_mask(slave)
                                            
                                            print(f"         Checking slave [{row},{col}]: '{slave_name}'")
                                            
                                            # Check if this is a kidney (case insensitive) but NOT SRF
                                            if ('kidney' in slave_name.lower() and 
                                                'srf' not in slave_name.lower() and 
                                                slave_mask is not None):
                                                
                                                # Verify mask matches image dimensions
                                                if slave_mask.shape == img_data.shape:
                                                    print(f"            ✅ Found kidney: '{slave_name}' with matching mask")
                                                    
                                                    # Normalize data
                                                    mri_normalized = normalize_data(img_data.astype(np.float32))
                                                    mask_normalized = (slave_mask > 0).astype(np.float32)  # Binary mask
                                                    
                                                    # Resize to standard size
                                                    target_size = (128, 128, 32)
                                                    mri_resized = resize_to_standard(mri_normalized, target_size)
                                                    mask_resized = resize_to_standard(mask_normalized, target_size)
                                                    
                                                    # Threshold mask after resizing
                                                    mask_resized = (mask_resized > 0.5).astype(np.float32)
                                                    
                                                    # Check if mask has reasonable coverage
                                                    mask_coverage = np.sum(mask_resized) / np.prod(mask_resized.shape)
                                                    
                                                    if mask_coverage > 0.001:  # At least 0.1% coverage
                                                        training_pairs.append({
                                                            'mri': mri_resized,
                                                            'mask': mask_resized,
                                                            'source_file': os.path.basename(mat_file_path),
                                                            'image_name': img_name,
                                                            'slave_name': slave_name,
                                                            'original_shape': img_data.shape,
                                                            'mask_coverage': mask_coverage
                                                        })
                                                        
                                                        print(f"               📊 Added training pair: coverage={mask_coverage*100:.2f}%")
                                                    else:
                                                        print(f"               ⚠️  Skipped: mask coverage too low ({mask_coverage*100:.3f}%)")
                                                else:
                                                    print(f"               ⚠️  Mask shape {slave_mask.shape} doesn't match image {img_data.shape}")
                                    except Exception as e:
                                        print(f"         ❌ Error processing slave [{row},{col}]: {e}")
                        else:
                            # Fallback for 1D arrays
                            for j in range(slaves.shape[0]):
                                try:
                                    slave = slaves[j, 0] if len(slaves.shape) > 1 else slaves[j]
                                    
                                    if slave.size > 0:  # Only process non-empty slaves
                                        # Extract slave name and mask
                                        slave_name = extract_slave_name(slave)
                                        slave_mask = extract_slave_mask(slave)
                                        
                                        print(f"         Checking slave {j}: '{slave_name}'")
                                        
                                        # Check if this is a kidney (case insensitive) but NOT SRF
                                        if ('kidney' in slave_name.lower() and 
                                            'srf' not in slave_name.lower() and 
                                            slave_mask is not None):
                                            
                                            # Same processing as above...
                                            if slave_mask.shape == img_data.shape:
                                                print(f"            ✅ Found kidney: '{slave_name}' with matching mask")
                                                
                                                # Normalize data
                                                mri_normalized = normalize_data(img_data.astype(np.float32))
                                                mask_normalized = (slave_mask > 0).astype(np.float32)
                                                
                                                # Resize to standard size
                                                target_size = (128, 128, 32)
                                                mri_resized = resize_to_standard(mri_normalized, target_size)
                                                mask_resized = resize_to_standard(mask_normalized, target_size)
                                                
                                                # Threshold mask after resizing
                                                mask_resized = (mask_resized > 0.5).astype(np.float32)
                                                
                                                # Check if mask has reasonable coverage
                                                mask_coverage = np.sum(mask_resized) / np.prod(mask_resized.shape)
                                                
                                                if mask_coverage > 0.001:
                                                    training_pairs.append({
                                                        'mri': mri_resized,
                                                        'mask': mask_resized,
                                                        'source_file': os.path.basename(mat_file_path),
                                                        'image_name': img_name,
                                                        'slave_name': slave_name,
                                                        'original_shape': img_data.shape,
                                                        'mask_coverage': mask_coverage
                                                    })
                                                    
                                                    print(f"               📊 Added training pair: coverage={mask_coverage*100:.2f}%")
                                                else:
                                                    print(f"               ⚠️  Skipped: mask coverage too low ({mask_coverage*100:.3f}%)")
                                            else:
                                                print(f"               ⚠️  Mask shape {slave_mask.shape} doesn't match image {img_data.shape}")
                                
                                except Exception as e:
                                    print(f"         ❌ Error processing slave {j}: {e}")
                    else:
                        print(f"      No slaves found")
                
            except Exception as e:
                print(f"   ❌ Error processing image {i}: {e}")
    
    return training_pairs

def main():
    """Create training data from all .mat files"""
    print("🎯 CREATING KIDNEY TRAINING DATA")
    print("=" * 50)
    
    # Define paths
    training_data_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
    output_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingKidneys\src\training\kidneyTrainingData"
    
    # Get all .mat files
    mat_files = []
    if os.path.exists(training_data_dir):
        for file in os.listdir(training_data_dir):
            if file.endswith('.mat'):
                mat_files.append(os.path.join(training_data_dir, file))
    
    print(f"📂 Found {len(mat_files)} .mat files")
    
    # Extract training data from all files
    all_training_pairs = []
    
    for mat_file in mat_files:
        try:
            pairs = extract_training_data_from_file(mat_file)
            all_training_pairs.extend(pairs)
        except Exception as e:
            print(f"❌ Error processing {mat_file}: {e}")
    
    print(f"\n📊 EXTRACTION SUMMARY")
    print("=" * 30)
    print(f"Total training pairs extracted: {len(all_training_pairs)}")
    
    if not all_training_pairs:
        print("❌ No valid training pairs extracted!")
        return
    
    # Save training data as numpy arrays
    for i, pair in enumerate(all_training_pairs):
        # Save MRI data
        mri_path = os.path.join(output_dir, f"mri_{i:03d}.npy")
        np.save(mri_path, pair['mri'])
        
        # Save mask data
        mask_path = os.path.join(output_dir, f"mask_{i:03d}.npy")
        np.save(mask_path, pair['mask'])
        
        print(f"✅ Saved pair {i:03d}: {pair['source_file']} - {pair['slave_name']}")
        print(f"   MRI: {mri_path}")
        print(f"   Mask: {mask_path}")
        print(f"   Shape: {pair['mri'].shape}, Coverage: {pair['mask_coverage']*100:.2f}%")
    
    # Save metadata
    metadata = {
        'total_pairs': len(all_training_pairs),
        'pairs': [
            {
                'index': i,
                'source_file': pair['source_file'],
                'image_name': pair['image_name'],
                'slave_name': pair['slave_name'],
                'original_shape': pair['original_shape'],
                'final_shape': pair['mri'].shape,
                'mask_coverage': pair['mask_coverage']
            }
            for i, pair in enumerate(all_training_pairs)
        ]
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Saved metadata: {metadata_path}")
    print("🎉 Training data extraction complete!")
    print(f"📂 All files saved in: {output_dir}")
    
    # Show summary statistics
    coverages = [p['mask_coverage'] for p in all_training_pairs]
    print(f"\n📈 COVERAGE STATISTICS:")
    print(f"   Min coverage: {min(coverages)*100:.3f}%")
    print(f"   Max coverage: {max(coverages)*100:.3f}%")
    print(f"   Mean coverage: {np.mean(coverages)*100:.3f}%")
    print(f"   Median coverage: {np.median(coverages)*100:.3f}%")

if __name__ == "__main__":
    main()
