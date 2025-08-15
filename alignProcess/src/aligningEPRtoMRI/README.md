# EPR-to-MRI Alignment System

This directory contains a complete pipeline for aligning EPR images to MRI images using AI-detected fiducials.

## Quick Start

### Option 1: Use the automated batch script (Recommended)

```batch
# Run with default training data
.\run.bat

# Run with your own data file
.\run.bat "C:\path\to\your\data.mat"
```

### Option 2: Use MATLAB directly

```matlab
% Run the complete pipeline interactively
create_alignment_pipeline()

% Or run the core alignment function
align_EPR_to_MRI()
```

## What it does

1. **Fiducial Detection**: Automatically runs AI fiducial detection if needed
2. **Coordinate Extraction**: Extracts 3D centroids from detected fiducial masks
3. **Alignment Calculation**: Computes optimal scaling, rotation, and translation
4. **Transformation Application**: Updates ArbuzGUI transformation matrices
5. **Report Generation**: Creates detailed alignment reports

## Files

### Core Scripts
- `run.bat` - Complete automated pipeline (recommended)
- `align_EPR_to_MRI.m` - Core alignment algorithm
- `create_alignment_pipeline.m` - Interactive MATLAB pipeline

### Analysis & Verification
- `verify_alignment_results.m` - Verify alignment accuracy
- `examine_transformation_matrices.m` - Examine transformation structure
- `debug_fiducials.m` - Debug fiducial extraction
- `generate_alignment_report.m` - Generate detailed reports

## Input Requirements

Your `.mat` file should contain:
- `images` cell array with at least:
  - One MRI image (reference)
  - One or more EPR images to align
- Each image should have proper `Name` field
- Compatible with ArbuzGUI format

## Output

The system creates a timestamped output directory containing:
- `aligned_data.mat` - Main result (load this in ArbuzGUI)
- `alignment_report.txt` - Detailed alignment statistics
- `summary.txt` - Processing summary
- `original_data.mat` - Copy of input data
- `fiducials_data.mat` - Copy of fiducials data

## Alignment Results

Typical results for EPR-to-MRI alignment:
- **Scale Factor**: ~6.4x (matches voxel size ratio: 0.6629mm/0.1mm)
- **Alignment Error**: ~14-15 voxels mean error
- **Fiducials Required**: Minimum 3 per EPR image

## Troubleshooting

### Common Issues

1. **"No fiducials found"**
   - Run fiducial detection first: `run.bat`
   - Check that your data has detectable fiducial markers

2. **"Not enough fiducials"**
   - Need at least 3 fiducials per EPR image
   - Check fiducial detection quality

3. **"MATLAB not available"**
   - Ensure MATLAB is installed and in PATH
   - Test with: `matlab -batch "fprintf('OK\n')"`

### File Structure

```
aligningEPRtoMRI/
├── run.bat                    # Main pipeline script
├── align_EPR_to_MRI.m        # Core alignment
├── create_alignment_pipeline.m # Interactive pipeline
├── verify_alignment_results.m # Verification
├── output/                   # Generated results
│   └── alignment_YYYYMMDD_HHMMSS/
│       ├── aligned_data.mat
│       ├── alignment_report.txt
│       └── summary.txt
└── README.md                 # This file
```

## Technical Details

### Transformation Algorithm
1. Extract fiducial centroids from detection masks
2. Calculate optimal rigid transformation (scaling + rotation + translation)
3. Apply transformation to ArbuzGUI A matrices
4. Preserve original coordinate system compatibility

### Coordinate Systems
- **MRI**: 350×350×18/40, 0.1mm spacing
- **EPR**: 64×64×64, 0.6629mm spacing
- **Alignment**: EPR → MRI coordinate system

## Examples

### Basic Usage
```batch
# Align default training data
cd C:\Users\ftmen\Documents\mrialign\alignProcess\src\aligningEPRtoMRI
.\run.bat
```

### Custom Data
```batch
# Align your own data
.\run.bat "C:\my_data\experiment_001.mat"
```

### MATLAB Verification
```matlab
% Load and verify results
data = load('output\alignment_20250814_140951\aligned_data.mat');
verify_alignment_results();
```

## Support

For questions or issues:
1. Check the generated `alignment_report.txt` for details
2. Run verification scripts to diagnose problems
3. Ensure input data is in proper ArbuzGUI format
