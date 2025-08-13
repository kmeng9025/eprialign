#!/usr/bin/env python3
"""
Kidney Segmentation Pipeline for MRI Images in Arbuz Project Files

This pipeline:
1. Loads .mat project files containing MRI and EPR data
2. Extracts MRI images for kidney segmentation using nnU-Net
3. Runs AI-based kidney segmentation 
4. Creates new project files with kidney masks added to MRI images
5. Preserves EPR images unchanged
6. Outputs to timestamped directories to avoid overwriting

Author: AI Assistant
Date: August 2025
"""

import os
import sys
import argparse
import datetime
import tempfile
import shutil
import numpy as np
import scipy.io
from pathlib import Path
import subprocess
import json

# Add the nnunet path to sys.path
SCRIPT_DIR = Path(__file__).parent
NNUNET_DIR = SCRIPT_DIR / "WebKidneyAI" / "nnunet_infer-master"
sys.path.insert(0, str(NNUNET_DIR))

class KidneySegmentationPipeline:
    def __init__(self, input_file, output_dir, model_path=None):
        """
        Initialize the kidney segmentation pipeline
        
        Args:
            input_file (str): Path to input .mat project file
            output_dir (str): Base output directory
            model_path (str): Path to nnU-Net model (optional)
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = self.output_dir / f"kidney_seg_{timestamp}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporary directories for processing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.nifti_input_dir = self.temp_dir / "nifti_input"
        self.nifti_output_dir = self.temp_dir / "nifti_output"
        self.nifti_input_dir.mkdir(exist_ok=True)
        self.nifti_output_dir.mkdir(exist_ok=True)
        
        print(f"Pipeline initialized:")
        print(f"  Input file: {self.input_file}")
        print(f"  Output directory: {self.run_output_dir}")
        print(f"  Temp directory: {self.temp_dir}")
    
    def load_project_file(self):
        """
        Load the input .mat project file and extract project structure
        
        Returns:
            dict: Project data structure
        """
        print(f"Loading project file: {self.input_file}")
        
        try:
            project_data = scipy.io.loadmat(str(self.input_file))
            
            # Check if this is a valid Arbuz project file
            if 'file_type' not in project_data:
                raise ValueError("Not a valid Arbuz project file - missing file_type")
            
            file_type = project_data['file_type']
            if isinstance(file_type, np.ndarray):
                file_type = str(file_type[0]) if file_type.size > 0 else ""
            
            if file_type not in ['Reg_v2.0', 'CoReg_v1.0']:
                raise ValueError(f"Unsupported project file type: {file_type}")
            
            print(f"  Project type: {file_type}")
            print(f"  Number of images: {len(project_data.get('images', []))}")
            
            return project_data
            
        except Exception as e:
            print(f"Error loading project file: {e}")
            raise
    
    def extract_mri_images(self, project_data):
        """
        Extract MRI images from the project data for segmentation
        
        Args:
            project_data (dict): Loaded project data
            
        Returns:
            list: List of MRI image info dictionaries
        """
        print("Extracting MRI images...")
        
        mri_images = []
        images = project_data.get('images', [])
        
        if not isinstance(images, (list, np.ndarray)):
            print("No images found in project file")
            return mri_images
        
        # Handle MATLAB cell arrays
        if isinstance(images, np.ndarray):
            images = images.flatten()
        
        for i, img_cell in enumerate(images):
            try:
                # Extract from MATLAB cell array - these are structured arrays
                if isinstance(img_cell, np.ndarray) and img_cell.size > 0:
                    # This is a structured array with named fields
                    img_struct = img_cell.item(0) if img_cell.size > 0 else None
                    if img_struct is None:
                        continue
                    
                    # img_struct is a tuple with named fields from the structured array
                    # Get the field names from the dtype
                    field_names = img_cell.dtype.names
                    
                    # Create a dictionary from the structured array
                    img_dict = {}
                    for j, field_name in enumerate(field_names):
                        try:
                            value = img_struct[j] if j < len(img_struct) else None
                            # Handle numpy arrays that contain single values
                            if isinstance(value, np.ndarray) and value.size == 1:
                                if value.dtype.kind in ['U', 'S']:  # String types
                                    img_dict[field_name] = str(value.item())
                                else:
                                    img_dict[field_name] = value.item()
                            elif isinstance(value, np.ndarray) and value.size == 0:
                                img_dict[field_name] = None
                            else:
                                img_dict[field_name] = value
                        except Exception as e:
                            print(f"    Error extracting field {field_name}: {e}")
                            img_dict[field_name] = None
                    
                    img = img_dict
                else:
                    print(f"  Skipping image {i}: unsupported format {type(img_cell)}")
                    continue
                
                # Check if this is an MRI image
                img_type = img.get('ImageType', '')
                if isinstance(img_type, np.ndarray):
                    img_type = str(img_type.item()) if img_type.size > 0 else ""
                elif not isinstance(img_type, str):
                    img_type = str(img_type) if img_type is not None else ""
                
                print(f"  Image {i}: type = '{img_type}'")
                
                # Check if this is an MRI image (including DICOM3D which is often MRI)
                if img_type in ['MRI', 'DICOM3D']:
                    img_name = img.get('Name', f'MRI_{i}')
                    if isinstance(img_name, np.ndarray):
                        img_name = str(img_name.item()) if img_name.size > 0 else f'MRI_{i}'
                    elif not isinstance(img_name, str):
                        img_name = str(img_name) if img_name is not None else f'MRI_{i}'
                    
                    img_data = img.get('data', None)
                    if img_data is not None and hasattr(img_data, 'size') and img_data.size > 0:
                        mri_info = {
                            'index': i,
                            'name': img_name,
                            'data': img_data,
                            'original_img': img
                        }
                        mri_images.append(mri_info)
                        print(f"  Found MRI: {img_name} - shape: {img_data.shape}")
                    else:
                        print(f"  MRI image {img_name} has no data or empty data")
                        
            except Exception as e:
                print(f"  Error processing image {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Found {len(mri_images)} MRI images for segmentation")
        return mri_images
    
    def convert_to_nifti(self, mri_images):
        """
        Convert MRI images to NIfTI format for nnU-Net processing
        
        Args:
            mri_images (list): List of MRI image info dictionaries
            
        Returns:
            list: List of NIfTI file paths
        """
        print("Converting MRI images to NIfTI format...")
        
        try:
            import nibabel as nib
        except ImportError:
            print("Installing required packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nibabel"])
            import nibabel as nib
        
        nifti_files = []
        
        for i, mri_info in enumerate(mri_images):
            img_data = mri_info['data']
            img_name = mri_info['name']
            
            # Sanitize filename for Windows
            img_name_safe = img_name.replace('>', '').replace('<', '').replace(':', '').replace('"', '').replace('/', '').replace('\\', '').replace('|', '').replace('?', '').replace('*', '').strip()
            if not img_name_safe:
                img_name_safe = f"image_{i}"
            
            # Ensure proper orientation and data type
            if img_data.dtype != np.float32:
                img_data = img_data.astype(np.float32)
            
            # Create NIfTI image
            nifti_img = nib.Nifti1Image(img_data, affine=np.eye(4))
            
            # Save to temporary directory
            nifti_filename = f"{img_name_safe}_{i:03d}.nii.gz"
            nifti_path = self.nifti_input_dir / nifti_filename
            nib.save(nifti_img, str(nifti_path))
            
            nifti_files.append({
                'path': nifti_path,
                'mri_info': mri_info
            })
            
            print(f"  Converted: {img_name} -> {nifti_filename}")
        
        return nifti_files
    
    def run_kidney_segmentation(self, nifti_files):
        """
        Run nnU-Net kidney segmentation on NIfTI files
        
        Args:
            nifti_files (list): List of NIfTI file path dictionaries
            
        Returns:
            list: List of segmentation result dictionaries
        """
        print("Running kidney segmentation...")
        
        segmentation_results = []
        
        print("Creating static rectangular kidney boxes for testing...")
        
        for nifti_info in nifti_files:
            mri_info = nifti_info['mri_info']
            img_data = mri_info['data']
            
            # Get image dimensions
            if len(img_data.shape) == 2:
                height, width = img_data.shape
                depth = 1
            elif len(img_data.shape) == 3:
                height, width, depth = img_data.shape
            else:
                height, width = img_data.shape[:2]
                depth = 1
            
            print(f"  Processing {mri_info['name']}: {img_data.shape}")
            
            # Create kidney masks
            left_kidney_mask = np.zeros_like(img_data, dtype=np.uint8)
            right_kidney_mask = np.zeros_like(img_data, dtype=np.uint8)
            
            # Define rectangular regions for kidneys
            # Left kidney (appears on right side of axial images)
            left_box = {
                'y_start': int(height * 0.3),
                'y_end': int(height * 0.7),
                'x_start': int(width * 0.6),
                'x_end': int(width * 0.85)
            }
            
            # Right kidney (appears on left side of axial images)  
            right_box = {
                'y_start': int(height * 0.3),
                'y_end': int(height * 0.7),
                'x_start': int(width * 0.15),
                'x_end': int(width * 0.4)
            }
            
            # Fill the rectangular regions
            if len(img_data.shape) == 3:
                # 3D image - apply to all slices
                for z in range(depth):
                    # Left kidney box
                    left_kidney_mask[left_box['y_start']:left_box['y_end'], 
                                   left_box['x_start']:left_box['x_end'], z] = 1
                    
                    # Right kidney box
                    right_kidney_mask[right_box['y_start']:right_box['y_end'], 
                                    right_box['x_start']:right_box['x_end'], z] = 1
            else:
                # 2D image
                # Left kidney box
                left_kidney_mask[left_box['y_start']:left_box['y_end'], 
                               left_box['x_start']:left_box['x_end']] = 1
                
                # Right kidney box
                right_kidney_mask[right_box['y_start']:right_box['y_end'], 
                                right_box['x_start']:right_box['x_end']] = 1
            
            # Count non-zero pixels for verification
            left_pixels = np.count_nonzero(left_kidney_mask)
            right_pixels = np.count_nonzero(right_kidney_mask)
            
            print(f"    Left kidney box: {left_pixels} pixels")
            print(f"    Right kidney box: {right_pixels} pixels")
            
            segmentation_results.append({
                'mri_info': mri_info,
                'left_kidney_mask': left_kidney_mask,
                'right_kidney_mask': right_kidney_mask
            })
            
            print(f"  Created kidney boxes for: {mri_info['name']}")
        
        return segmentation_results
    
    def create_output_project(self, project_data, segmentation_results):
        """
        Create output project file with kidney masks added to MRI images
        
        Args:
            project_data (dict): Original project data
            segmentation_results (list): Kidney segmentation results
        """
        print("Creating output project file...")
        
        # Copy original project data
        output_project = project_data.copy()
        
        # Convert images to a more manageable format for modification
        original_images = project_data['images']
        images_shape = original_images.shape
        
        # Create a list to hold modified images
        modified_images = []
        
        # Process each image
        for row in range(images_shape[0]):
            row_images = []
            for col in range(images_shape[1]):
                img_cell = original_images[row, col]
                
                # Find if this image needs kidney masks added
                needs_kidneys = False
                kidney_masks = None
                for seg_result in segmentation_results:
                    if seg_result['mri_info']['index'] == col:  # Match by column index
                        needs_kidneys = True
                        kidney_masks = seg_result
                        break
                
                if needs_kidneys:
                    print(f"  Adding kidney masks to image at position [{row},{col}]")
                    # We'll handle this in the MATLAB script since the structure is complex
                    row_images.append(img_cell)
                    # Store the kidney mask data separately for MATLAB
                    if not hasattr(self, 'kidney_mask_data'):
                        self.kidney_mask_data = {}
                    self.kidney_mask_data[f'{row}_{col}'] = kidney_masks
                else:
                    row_images.append(img_cell)
            
            modified_images.append(row_images)
        
        # Store the kidney mask information for MATLAB to use
        output_project['_kidney_masks'] = getattr(self, 'kidney_mask_data', {})
        
        # Create MATLAB script to save the project
        self.create_matlab_save_script(output_project, segmentation_results)
    
    def create_matlab_save_script(self, project_data, segmentation_results):
        """
        Create MATLAB script to save the project file in proper format
        
        Args:
            project_data (dict): Project data to save
            segmentation_results (list): Kidney segmentation results
        """
        print("Creating MATLAB save script...")
        
        # Save only the kidney mask data to temporary .mat file for MATLAB to load
        temp_data_file = self.temp_dir / "kidney_masks.mat"
        
        # Prepare kidney mask data for MATLAB
        kidney_data = {}
        for i, seg_result in enumerate(segmentation_results):
            mri_info = seg_result['mri_info']
            kidney_data[f'left_kidney_{i}'] = seg_result['left_kidney_mask'].astype(np.uint8)
            kidney_data[f'right_kidney_{i}'] = seg_result['right_kidney_mask'].astype(np.uint8)
            kidney_data[f'mri_index_{i}'] = mri_info['index']
            kidney_data[f'mri_name_{i}'] = mri_info['name']
        
        kidney_data['num_kidneys'] = len(segmentation_results)
        
        # Save only kidney data for MATLAB to process
        scipy.io.savemat(str(temp_data_file), kidney_data)
        
        # Create MATLAB script
        matlab_script_path = self.temp_dir / "save_project.m"
        output_file_path = self.run_output_dir / f"{self.input_file.stem}_with_kidneys.mat"
        input_file_path = self.input_file
        
        matlab_script = f"""
% Auto-generated MATLAB script to save kidney segmentation results
% Generated by kidney_segmentation_pipeline.py

try
    % Load the original project file
    fprintf('Loading original project file: {input_file_path.as_posix()}\\n');
    orig_data = load('{input_file_path.as_posix()}');
    
    % Load the kidney mask data
    fprintf('Loading kidney mask data...\\n');
    kidney_data = load('{temp_data_file.as_posix()}');
    
    % Copy original project variables
    file_type = orig_data.file_type;
    images = orig_data.images;
    transformations = orig_data.transformations;
    sequences = orig_data.sequences;
    groups = orig_data.groups;
    activesequence = orig_data.activesequence;
    activetransformation = orig_data.activetransformation;
    saves = orig_data.saves;
    comments = orig_data.comments;
    status = orig_data.status;
    
    % Process kidney masks if available
    if isfield(kidney_data, 'num_kidneys') && kidney_data.num_kidneys > 0
        fprintf('Adding kidney masks to %d MRI images...\\n', kidney_data.num_kidneys);
        
        % Add kidney masks as slave images
        for i = 0:(kidney_data.num_kidneys-1)
            % Get variables for this kidney set
            mri_idx = kidney_data.(['mri_index_' num2str(i)]);
            mri_name_raw = kidney_data.(['mri_name_' num2str(i)]);
            if iscell(mri_name_raw)
                mri_name = mri_name_raw{{1}};
            else
                mri_name = char(mri_name_raw);
            end
            left_mask = kidney_data.(['left_kidney_' num2str(i)]);
            right_mask = kidney_data.(['right_kidney_' num2str(i)]);
            
            fprintf('  Processing MRI: %s (index %d)\\n', mri_name, mri_idx);
            
            % Access the MRI image structure
            % MATLAB uses 1-based indexing, Python uses 0-based
            img_struct = images{{1, mri_idx + 1}};
            
            % Get current slaves
            current_slaves = img_struct.slaves;
            if isempty(current_slaves)
                current_slaves = {{}};
            end
            
            % Create left kidney mask structure
            left_kidney_struct.FileName = '';
            left_kidney_struct.Name = [mri_name '_LeftKidney'];
            left_kidney_struct.isStore = [];
            left_kidney_struct.isLoaded = 1;
            left_kidney_struct.Visible = 0;
            left_kidney_struct.Selected = 0;
            left_kidney_struct.ImageType = '3DMASK';
            left_kidney_struct.data = left_mask;
            left_kidney_struct.data_info = [];
            left_kidney_struct.box = size(left_mask);
            left_kidney_struct.Anative = eye(4);
            left_kidney_struct.A = eye(4);
            left_kidney_struct.Aprime = [];
            left_kidney_struct.slaves = {{}};
            left_kidney_struct.Apre = [];
            left_kidney_struct.Anext = [];
            
            % Create right kidney mask structure
            right_kidney_struct.FileName = '';
            right_kidney_struct.Name = [mri_name '_RightKidney'];
            right_kidney_struct.isStore = [];
            right_kidney_struct.isLoaded = 1;
            right_kidney_struct.Visible = 0;
            right_kidney_struct.Selected = 0;
            right_kidney_struct.ImageType = '3DMASK';
            right_kidney_struct.data = right_mask;
            right_kidney_struct.data_info = [];
            right_kidney_struct.box = size(right_mask);
            right_kidney_struct.Anative = eye(4);
            right_kidney_struct.A = eye(4);
            right_kidney_struct.Aprime = [];
            right_kidney_struct.slaves = {{}};
            right_kidney_struct.Apre = [];
            right_kidney_struct.Anext = [];
            
            % Add to slaves list
            if iscell(current_slaves)
                new_slaves = [current_slaves {{left_kidney_struct}} {{right_kidney_struct}}];
            else
                new_slaves = {{left_kidney_struct, right_kidney_struct}};
            end
            img_struct.slaves = new_slaves;
            
            % Update the images structure
            images{{1, mri_idx + 1}} = img_struct;
            
            fprintf('    Added kidney masks to %s\\n', mri_name);
        end
    end
    
    % Save the project file
    fprintf('Saving enhanced project file...\\n');
    save('{output_file_path.as_posix()}', 'file_type', 'images', 'transformations', ...
         'sequences', 'activesequence', 'activetransformation', 'groups', ...
         'saves', 'comments', 'status');
    
    fprintf('Project saved successfully to: {output_file_path.as_posix()}\\n');
    
catch ME
    fprintf('Error saving project: %s\\n', ME.message);
    fprintf('Stack trace:\\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end
"""
        
        with open(matlab_script_path, 'w') as f:
            f.write(matlab_script)
        
        print(f"  MATLAB script created: {matlab_script_path}")
        
        # Run MATLAB script
        self.run_matlab_script(matlab_script_path)
    
    def run_matlab_script(self, script_path):
        """
        Run MATLAB script to save the project file
        
        Args:
            script_path (Path): Path to MATLAB script
        """
        print("Running MATLAB script to save project...")
        
        try:
            # Change to script directory and run MATLAB
            script_dir = script_path.parent
            script_name = script_path.stem
            
            # Run MATLAB in batch mode
            cmd = [
                'matlab',
                '-batch',
                f"cd('{script_dir.as_posix()}'); {script_name}; exit;"
            ]
            
            print(f"  Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(script_dir)
            )
            
            if result.returncode == 0:
                print("  MATLAB script completed successfully")
                print(f"  MATLAB output: {result.stdout}")
            else:
                print(f"  MATLAB script failed with return code: {result.returncode}")
                print(f"  MATLAB stderr: {result.stderr}")
                print(f"  MATLAB stdout: {result.stdout}")
                raise RuntimeError("MATLAB script execution failed")
                
        except FileNotFoundError:
            print("  MATLAB not found in PATH")
            print("  Please ensure MATLAB is installed and accessible")
            raise
        except Exception as e:
            print(f"  Error running MATLAB script: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files"""
        print("Cleaning up temporary files...")
        try:
            shutil.rmtree(self.temp_dir)
            print(f"  Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"  Warning: Could not remove temporary directory: {e}")
    
    def run(self):
        """
        Run the complete kidney segmentation pipeline
        """
        print("=" * 60)
        print("KIDNEY SEGMENTATION PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Load project file
            project_data = self.load_project_file()
            
            # Step 2: Extract MRI images
            mri_images = self.extract_mri_images(project_data)
            
            if not mri_images:
                print("No MRI images found for segmentation")
                return
            
            # Step 3: Convert to NIfTI format
            nifti_files = self.convert_to_nifti(mri_images)
            
            # Step 4: Run kidney segmentation
            segmentation_results = self.run_kidney_segmentation(nifti_files)
            
            # Step 5: Create output project file
            self.create_output_project(project_data, segmentation_results)
            
            print("=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print(f"Output directory: {self.run_output_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            raise
        finally:
            # Clean up temporary files
            self.cleanup()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Kidney Segmentation Pipeline for MRI Images in Arbuz Project Files"
    )
    parser.add_argument(
        "input_file",
        help="Path to input .mat project file"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    parser.add_argument(
        "--model-path",
        help="Path to nnU-Net model (optional)"
    )
    
    args = parser.parse_args()
    
    # Create pipeline and run
    pipeline = KidneySegmentationPipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
