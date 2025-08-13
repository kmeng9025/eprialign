import os
import numpy as np
import scipy.io as sio
from glob import glob
from scipy import ndimage

# Set your input and output directories
INPUT_DIR = '../data/training'  # Folder where your .mat files are stored
OUTPUT_DIR = '../data/preprocessed_fiducials'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target dimensions for standardization
TARGET_SIZE = (64, 64, 64)

def resize_3d_volume(volume, target_size):
    """Resize a 3D volume to target size using scipy interpolation"""
    zoom_factors = [t/s for t, s in zip(target_size, volume.shape)]
    return ndimage.zoom(volume, zoom_factors, order=1)

def normalize_image(image):
    """Normalize image to [0, 1] range"""
    image = image.astype(np.float32)
    min_val, max_val = image.min(), image.max()
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image

def extract_fiducials_from_entry(entry):
    """Extract image data and fiducial masks from a single image entry"""
    if not hasattr(entry, 'data') or not hasattr(entry, 'slaves'):
        return None, None
    
    # Get the main image data
    image_data = entry.data
    if not hasattr(image_data, 'shape') or len(image_data.shape) != 3:
        return None, None
    
    # Extract fiducial masks from slaves
    slaves_list = entry.slaves.flatten() if isinstance(entry.slaves, np.ndarray) else [entry.slaves]
    fiducial_masks = []
    fiducial_names = []
    
    for slave in slaves_list:
        if hasattr(slave, 'Name') and hasattr(slave, 'data'):
            name = str(slave.Name)
            # Check if this is a fiducial (contains FID or Fiducial)
            if ('FID' in name.upper() or 'FIDUCIAL' in name.upper()) and hasattr(slave.data, 'shape'):
                if len(slave.data.shape) == 3:  # Only 3D data
                    fiducial_masks.append(slave.data.astype(np.uint8))
                    fiducial_names.append(name)
    
    if not fiducial_masks:
        return None, None
    
    return image_data, fiducial_masks, fiducial_names

# Lists to hold data
images = []
masks = []
metadata = []

# Search for .mat files in the input directory
mat_files = glob(os.path.join(INPUT_DIR, '*.mat'))
print(f"Found {len(mat_files)} .mat files")

for file_idx, file in enumerate(mat_files):
    print(f"\nProcessing file {file_idx+1}/{len(mat_files)}: {os.path.basename(file)}")
    
    try:
        data = sio.loadmat(file, struct_as_record=False, squeeze_me=True)
        
        if 'images' not in data:
            print(f"  Skipping - no 'images' field found")
            continue
            
        image_entries = data['images']
        
        for entry_idx, entry in enumerate(image_entries):
            if not hasattr(entry, 'Name'):
                continue
                
            entry_name = str(entry.Name)
            print(f"  Processing entry: {entry_name}")
            
            # Extract image and fiducials
            result = extract_fiducials_from_entry(entry)
            if result is None or result[0] is None:
                print(f"    No valid data found")
                continue
                
            image_data, fiducial_masks, fiducial_names = result
            print(f"    Found image shape: {image_data.shape}")
            print(f"    Found {len(fiducial_masks)} fiducials: {fiducial_names}")
            
            # Resize image to target size
            resized_image = resize_3d_volume(image_data, TARGET_SIZE)
            normalized_image = normalize_image(resized_image)
            
            # Process each fiducial mask
            for fid_idx, (mask, fid_name) in enumerate(zip(fiducial_masks, fiducial_names)):
                # Resize mask to target size
                resized_mask = resize_3d_volume(mask.astype(np.float32), TARGET_SIZE)
                # Threshold to binary (important after resize)
                binary_mask = (resized_mask > 0.5).astype(np.float32)
                
                # Only keep if mask has some positive pixels
                if binary_mask.sum() > 0:
                    images.append(normalized_image[np.newaxis, ...])  # Add channel dim
                    masks.append(binary_mask[np.newaxis, ...])  # Add channel dim
                    
                    metadata.append({
                        'file': os.path.basename(file),
                        'entry_name': entry_name,
                        'fiducial_name': fid_name,
                        'original_image_shape': image_data.shape,
                        'original_mask_shape': mask.shape,
                        'mask_volume': binary_mask.sum()
                    })
                    
                    print(f"      Added {fid_name} (mask volume: {binary_mask.sum():.0f} voxels)")
                else:
                    print(f"      Skipped {fid_name} (empty mask)")

    except Exception as e:
        print(f"  Error processing {file}: {e}")
        continue

# Convert to numpy arrays
if len(images) == 0:
    print("❌ No valid fiducial data found!")
    exit(1)

print(f"\n📊 Dataset Summary:")
print(f"  Total samples: {len(images)}")
print(f"  Image shape: {images[0].shape}")
print(f"  Mask shape: {masks[0].shape}")

# Convert to arrays
X = np.stack(images, axis=0)  # Shape: (N, 1, 64, 64, 64)
Y = np.stack(masks, axis=0)   # Shape: (N, 1, 64, 64, 64)

print(f"  Final X shape: {X.shape}")
print(f"  Final Y shape: {Y.shape}")

# Save the preprocessed data
np.save(os.path.join(OUTPUT_DIR, 'X_fiducials.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'Y_fiducials.npy'), Y)

# Save metadata
metadata_file = os.path.join(OUTPUT_DIR, 'fiducial_metadata.txt')
with open(metadata_file, 'w') as f:
    f.write("Fiducial Dataset Metadata\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total samples: {len(metadata)}\n")
    f.write(f"Target size: {TARGET_SIZE}\n\n")
    
    for i, meta in enumerate(metadata):
        f.write(f"Sample {i+1}:\n")
        f.write(f"  File: {meta['file']}\n")
        f.write(f"  Entry: {meta['entry_name']}\n")
        f.write(f"  Fiducial: {meta['fiducial_name']}\n")
        f.write(f"  Original image shape: {meta['original_image_shape']}\n")
        f.write(f"  Original mask shape: {meta['original_mask_shape']}\n")
        f.write(f"  Mask volume: {meta['mask_volume']:.0f} voxels\n\n")

print(f"✅ Preprocessing complete!")
print(f"   Saved X_fiducials.npy and Y_fiducials.npy to {OUTPUT_DIR}")
print(f"   Saved metadata to {metadata_file}")

# Print some statistics
mask_volumes = [meta['mask_volume'] for meta in metadata]
print(f"\n📈 Fiducial Mask Statistics:")
print(f"   Mean volume: {np.mean(mask_volumes):.1f} voxels")
print(f"   Std volume: {np.std(mask_volumes):.1f} voxels")
print(f"   Min volume: {np.min(mask_volumes):.0f} voxels")
print(f"   Max volume: {np.max(mask_volumes):.0f} voxels")
