import os
import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Path to a training .mat file (edit as needed)
TRAINING_DIR = '../../../data/training'

# Find a .mat file to inspect
mat_files = [f for f in os.listdir(TRAINING_DIR) if f.endswith('.mat')]
if not mat_files:
    print('No .mat files found in', TRAINING_DIR)
    sys.exit(1)

mat_file = os.path.join(TRAINING_DIR, mat_files[0])
print(f'Inspecting file: {mat_file}')

data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)

if 'images' not in data:
    print('No "images" field in file!')
    sys.exit(1)

images = data['images']
print(f'File contains {len(images)} images:')

# Print all image names, shapes, Master/Slave info
for idx, img in enumerate(images):
    name = ''
    if hasattr(img, 'Name'):
        if isinstance(img.Name, str):
            name = img.Name
        else:
            try:
                name = ''.join(chr(c) for c in img.Name.flatten() if c != 0)
            except:
                name = f'img_{idx}'
    shape = getattr(img, 'data', np.array([])).shape if hasattr(img, 'data') else None
    master = getattr(img, 'Master', None)
    slave = getattr(img, 'Slave', None)
    print(f'  [{idx}] {name} {shape} | Master: {master} | Slave: {slave}')

# Find all MRIs
mri_indices = []
for idx, img in enumerate(images):
    name = ''
    if hasattr(img, 'Name'):
        if isinstance(img.Name, str):
            name = img.Name
        else:
            try:
                name = ''.join(chr(c) for c in img.Name.flatten() if c != 0)
            except:
                name = f'img_{idx}'
    if 'mri' in name.lower() and hasattr(img, 'data') and len(img.data.shape) == 3:
        mri_indices.append(idx)

print('\nMRIs found at indices:', mri_indices)

# For each MRI, list all slaves
for mri_idx in mri_indices:
    print(f'\n--- MRI index {mri_idx} ---')
    for idx, img in enumerate(images):
        master = getattr(img, 'Master', None)
        if master == mri_idx:
            name = ''
            if hasattr(img, 'Name'):
                if isinstance(img.Name, str):
                    name = img.Name
                else:
                    try:
                        name = ''.join(chr(c) for c in img.Name.flatten() if c != 0)
                    except:
                        name = f'img_{idx}'
            shape = getattr(img, 'data', np.array([])).shape if hasattr(img, 'data') else None
            arr = getattr(img, 'data', None)
            print(f'  Slave [{idx}] {name} {shape}')
            if arr is not None and isinstance(arr, np.ndarray):
                print(f'    min: {arr.min()}, max: {arr.max()}, unique: {np.unique(arr).size}')
                # Optionally save a slice for visual inspection
                if arr.ndim == 3:
                    plt.imsave(f'debug_mask_{mri_idx}_{idx}.png', arr[:,:,arr.shape[2]//2], cmap="gray")
                    print(f'    Saved middle slice as debug_mask_{mri_idx}_{idx}.png')
