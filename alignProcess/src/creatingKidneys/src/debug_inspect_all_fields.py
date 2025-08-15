import os
import sys
import scipy.io as sio
import numpy as np

TRAINING_DIR = '../../../data/training'
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

for idx, img in enumerate(images):
    print(f'\n--- Image {idx} ---')
    print(f'Type: {type(img)}')
    # Print all attributes/fields
    if hasattr(img, '__dict__'):
        for key, value in img.__dict__.items():
            print(f'  {key}: type={type(value)}')
            if isinstance(value, np.ndarray):
                print(f'    shape={value.shape}, dtype={value.dtype}, min={value.min() if value.size else "n/a"}, max={value.max() if value.size else "n/a"}')
            else:
                print(f'    value={value}')
    else:
        # If not a custom object, print as much as possible
        print(f'  dir: {dir(img)}')
        print(f'  str: {str(img)}')
