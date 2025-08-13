# Kidney Segmentation Pipeline

This pipeline performs AI-based kidney segmentation on MRI images contained in Arbuz project files (.mat format).

## Overview

The pipeline:

1. Loads .mat project files containing MRI and EPR data
2. Extracts MRI images for kidney segmentation using nnU-Net
3. Runs AI-based kidney segmentation to identify left and right kidneys
4. Creates new project files with kidney masks added as slave images to MRI images
5. Preserves EPR images unchanged
6. Outputs to timestamped directories to avoid overwriting previous runs

## File Structure

```
src/
├── kidney_segmentation_pipeline.py    # Main pipeline script
├── test_pipeline.py                   # Test script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── WebKidneyAI/                       # Pre-trained AI model directory
    └── nnunet_infer-master/           # nnU-Net inference code
```

## Requirements

### Software Requirements

- Python 3.7 or higher
- MATLAB (for saving output project files)
- Git (for cloning dependencies if needed)

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- numpy, scipy (scientific computing)
- nibabel (NIfTI image format support)
- SimpleITK (medical image processing)
- torch (PyTorch for nnU-Net)
- scikit-image (image processing)

## Usage

### Command Line Interface

Run the pipeline from the command line:

```bash
python kidney_segmentation_pipeline.py <input_file> [options]
```

**Arguments:**

- `input_file`: Path to input .mat project file

**Options:**

- `--output-dir`: Output directory for results (default: ./output)
- `--model-path`: Path to nnU-Net model (optional)

**Example:**

```bash
python kidney_segmentation_pipeline.py ../../data/training/withoutROIwithMRI.mat --output-dir ../../data/inference
```

### Test Script

Run the test script to test the pipeline with the provided test data:

```bash
python test_pipeline.py
```

This will:

- Use the test file: `../../data/training/withoutROIwithMRI.mat`
- Output results to: `../../data/inference/kidney_seg_YYYYMMDD_HHMMSS/`

### Programmatic Usage

You can also use the pipeline in your own Python scripts:

```python
from kidney_segmentation_pipeline import KidneySegmentationPipeline

# Create pipeline
pipeline = KidneySegmentationPipeline(
    input_file="path/to/input.mat",
    output_dir="path/to/output"
)

# Run segmentation
pipeline.run()
```

## Output

The pipeline creates a timestamped output directory containing:

```
kidney_seg_YYYYMMDD_HHMMSS/
└── <input_filename>_with_kidneys.mat    # Output project file
```

### Output Project File Structure

The output .mat file has the same structure as the input file, with kidney masks added:

- **MRI Images**: Original MRI images are preserved
- **Kidney Masks**: Two new slave images are added to each MRI image:
  - `<MRI_name>_LeftKidney`: Binary mask for left kidney
  - `<MRI_name>_RightKidney`: Binary mask for right kidney
- **EPR Images**: Preserved unchanged from input file
- **Project Metadata**: All transformations, sequences, etc. preserved

### Viewing Results

Open the output .mat file in ArbuzGUI to view and inspect the kidney segmentation results:

1. Launch MATLAB
2. Navigate to the Arbuz2.0 directory
3. Run `ArbuzGUI`
4. Open the output project file
5. The kidney masks will appear as slave images under each MRI image

## AI Model

### Current Implementation

The current implementation uses a placeholder segmentation algorithm that creates simple ellipsoidal regions as dummy kidney masks. This is intended as a framework that can be replaced with actual nnU-Net inference.

### nnU-Net Integration

To use the actual nnU-Net model:

1. Ensure the nnU-Net model is properly trained and available
2. Modify the `run_kidney_segmentation()` method in `kidney_segmentation_pipeline.py`
3. Replace the dummy segmentation code with actual nnU-Net inference calls

The code structure is designed to easily accommodate the actual nnU-Net model when available.

## Project File Format

### Input Format

The pipeline expects Arbuz project files (.mat) with the following structure:

```matlab
file_type: 'Reg_v2.0' or 'CoReg_v1.0'
images: {1×N cell}  % Cell array of image structures
  └── Name: string           % Image name
  └── ImageType: string      % 'MRI', 'EPR', etc.
  └── data: matrix          % Image data
  └── slaves: {1×M cell}    % Slave images (optional)
transformations: {1×P cell}   % Transformation data
sequences: {1×Q cell}        % Sequence data
% ... other project metadata
```

### MRI Image Structure

MRI images in the project should have:

- `ImageType`: 'MRI'
- `data`: 2D or 3D numerical array containing image data
- `Name`: Descriptive name for the image

## Troubleshooting

### Common Issues

1. **MATLAB not found**: Ensure MATLAB is installed and accessible from command line
2. **Missing dependencies**: Install required Python packages using `pip install -r requirements.txt`
3. **Memory issues**: For large images, ensure sufficient RAM is available
4. **File permissions**: Ensure write permissions for output directory

### Debug Mode

For debugging, you can modify the pipeline to preserve temporary files by commenting out the `cleanup()` call in the `run()` method.

## Future Enhancements

1. **Real nnU-Net Integration**: Replace dummy segmentation with trained model
2. **GPU Support**: Add CUDA support for faster inference
3. **Batch Processing**: Support for processing multiple project files
4. **Quality Control**: Add segmentation quality metrics and validation
5. **Visualization**: Add intermediate result visualization options

## License

This code is provided as-is for research and educational purposes.

## Contact

For questions or issues with this pipeline, please refer to the documentation or contact the development team.
