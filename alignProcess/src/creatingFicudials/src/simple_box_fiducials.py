import numpy as np
import scipy.io as sio
import os
import subprocess
import argparse
from datetime import datetime
import torch
from scipy import ndimage

# Import your trained model
from fiducial_model import FiducialUNet3D

def resize_3d_volume(volume, target_size):
    """Resize a 3D volume to target size using scipy interpolation"""
    zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
    return ndimage.zoom(volume, zoom_factors, order=1)

def normalize_image(image):
    """Normalize image to [0, 1] range"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(image)

def load_model(model_path, device):
    """Load the trained fiducial detection model"""
    model = FiducialUNet3D()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded AI model from {model_path}")
            if 'best_val_loss' in checkpoint:
                print(f"   Best validation loss: {checkpoint['best_val_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded AI model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.to(device)
    model.eval()
    return model

def create_simple_box_fiducials(image_shape, image_data=None, model=None, device=None, threshold=0.5, num_boxes=2, box_size=3):
    """
    Create fiducials using AI detection or fallback to simple boxes
    
    Args:
        image_shape: Shape of the 3D image (depth, height, width)
        image_data: The actual 3D image data for AI processing (optional)
        model: Trained AI model (optional)
        device: PyTorch device (optional)
        threshold: AI detection threshold (optional)
        num_boxes: Number of box fiducials to create if AI fails
        box_size: Size of each box fiducial if AI fails
    
    Returns:
        fiducial_mask: Binary mask with detected/created fiducials
        fiducial_count: Number of fiducials found/created
    """
    
    # Try AI detection first if model and data are provided
    if model is not None and image_data is not None and device is not None:
        try:
            print(f"  🤖 Using AI model for fiducial detection...")
            return create_ai_fiducials(image_data, model, device, threshold)
        except Exception as e:
            print(f"  ⚠️  AI detection failed: {e}")
            print(f"  📦 Falling back to static box creation...")
    
    # Fallback to static box creation
    print(f"  📦 Creating {num_boxes} static box fiducials...")
    mask = np.zeros(image_shape, dtype=bool)  # Use logical/boolean type like real fiducials
    
    depth, height, width = image_shape
    
    # Create 2 boxes at simple positions
    positions = [
        (depth//3, height//3, width//3),      # Box 1: Front-left
        (2*depth//3, 2*height//3, 2*width//3), # Box 2: Back-right
    ]
    
    fiducial_count = 0
    for i, (z, y, x) in enumerate(positions[:num_boxes]):
        # Make sure the box fits within the image
        z_start = max(0, z - box_size//2)
        z_end = min(depth, z + box_size//2 + 1)
        y_start = max(0, y - box_size//2)
        y_end = min(height, y + box_size//2 + 1)
        x_start = max(0, x - box_size//2)
        x_end = min(width, x + box_size//2 + 1)
        
        # Create the box
        mask[z_start:z_end, y_start:y_end, x_start:x_end] = True  # Use boolean True
        fiducial_count += 1
        
        print(f"    📦 Created box {i+1} at ({z}, {y}, {x}) with size {box_size}x{box_size}x{box_size}")
    
    return mask, fiducial_count

def create_ai_fiducials(image_data, model, device, threshold=0.5, target_size=(64, 64, 64)):
    """
    Use AI model to detect fiducials in 3D image data
    
    Args:
        image_data: 3D numpy array of image data
        model: Trained AI model
        device: PyTorch device
        threshold: Detection threshold
        target_size: Target size for model input
    
    Returns:
        fiducial_mask: Binary mask with AI-detected fiducials
        fiducial_count: Number of fiducials detected
    """
    original_shape = image_data.shape
    
    print(f"    🔍 AI processing volume: {original_shape}")
    
    # Normalize the entire volume
    normalized_image = normalize_image(image_data)
    
    # Resize to model input size if needed
    if original_shape != target_size:
        resized_image = resize_3d_volume(normalized_image, target_size)
    else:
        resized_image = normalized_image
    
    # Prepare tensor
    input_tensor = torch.tensor(resized_image[np.newaxis, np.newaxis, ...], 
                               dtype=torch.float32).to(device)
    
    # Predict on entire volume
    with torch.no_grad():
        output = model(input_tensor)
        confidence_map = output.cpu().numpy()[0, 0]  # Remove batch and channel dims
    
    # Binary prediction
    binary_prediction = (confidence_map > threshold).astype(bool)
    
    # Resize back to original size if needed
    if original_shape != target_size:
        # Resize binary prediction back to original size
        binary_prediction_resized = resize_3d_volume(binary_prediction.astype(float), original_shape)
        binary_prediction = (binary_prediction_resized > 0.5).astype(bool)
    
    # Count connected components as individual fiducials
    from scipy.ndimage import label
    labeled_fiducials, num_fiducials = label(binary_prediction)
    
    # Filter out very small detections (likely noise)
    min_volume = 5  # Minimum voxels per fiducial
    filtered_mask = np.zeros_like(binary_prediction, dtype=bool)
    actual_count = 0
    
    for i in range(1, num_fiducials + 1):
        fiducial_region = (labeled_fiducials == i)
        if np.sum(fiducial_region) >= min_volume:
            filtered_mask |= fiducial_region
            actual_count += 1
            
            # Get centroid for reporting
            coords = np.where(fiducial_region)
            centroid = (np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2]))
            volume = np.sum(fiducial_region)
            print(f"    🎯 AI detected fiducial {actual_count} at {centroid} (volume: {volume} voxels)")
    
    print(f"    ✅ AI found {actual_count} fiducials (threshold: {threshold})")
    
    return filtered_mask, actual_count

def save_box_data_for_arbuz(results, output_dir, project_name):
    """
    Save box fiducial data for ArbuzGUI integration
    """
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_output_dir = os.path.join(output_dir, f"box_fiducials_{project_name}_{timestamp}")
    os.makedirs(unique_output_dir, exist_ok=True)
    
    # Prepare data for MATLAB/ArbuzGUI
    image_names = [result['entry_name'] for result in results]
    num_fiducials = [result['num_fiducials'] for result in results]
    
    # Convert numpy boolean arrays to MATLAB-compatible format
    fiducial_masks = []
    for result in results:
        mask = result['fiducial_mask']
        # Ensure proper 3D boolean array for MATLAB
        matlab_mask = mask.astype(bool)  # Ensure boolean type
        fiducial_masks.append(matlab_mask)
    
    box_data = {
        'image_names': image_names,  # Don't wrap in np.array to avoid object array issues
        'fiducial_masks': fiducial_masks,  # List of 3D boolean arrays
        'num_fiducials': num_fiducials,
        'project_name': project_name,
        'timestamp': timestamp
    }
    
    # Save as .mat file
    data_file_path = os.path.join(unique_output_dir, 'box_fiducial_data.mat')
    sio.savemat(data_file_path, box_data, format='5')
    
    print(f"💾 Box fiducial data saved to: {data_file_path}")
    print(f"   Images processed: {len(image_names)}")
    print(f"   Total boxes created: {sum(num_fiducials)}")
    
    return data_file_path, unique_output_dir

def create_arbuz_integration_script(data_file_path, original_project_path, output_dir, script_name):
    """
    Create a MATLAB script that uses ArbuzGUI's native functions to add box fiducials
    """
    script_path = os.path.join(output_dir, f'{script_name}.m')
    
    # Use absolute paths
    data_file_name = os.path.basename(data_file_path)
    original_project_abs = os.path.abspath(original_project_path).replace('\\', '/')
    output_project_name = f'{os.path.splitext(os.path.basename(original_project_path))[0]}_with_boxes.mat'
    
    matlab_script = f'''function {script_name}()
% Manually add fiducials as slaves without using GUI functions
% This creates the exact same structure as existing ArbuzGUI fiducial files

try
    fprintf('Loading project and fiducial data...\\n');
    
    % Load the original project
    original_file = '{original_project_abs}';
    project_data = load(original_file);
    
    % Load our fiducial data
    box_data = load('{data_file_name}');
    
    fprintf('Original project: %d images\\n', length(project_data.images));
    fprintf('Box data: %d images with fiducials\\n', length(box_data.image_names));
    
    % Process each image and add fiducials as slaves
    total_added = 0;
    
    for i = 1:length(project_data.images)
        current_image = project_data.images{{i}};
        image_name = current_image.Name;
        fprintf('Processing image: %s\\n', image_name);
        
        % Find matching box data
        box_idx = -1;
        for j = 1:size(box_data.image_names, 1)
            % Handle character array format from scipy.io.savemat
            if size(box_data.image_names, 2) > 1
                % Character array format [num_images x max_name_length]
                stored_name = strtrim(box_data.image_names(j, :));
            else
                % Cell array format (if it were preserved)
                stored_name = box_data.image_names{{j}};
            end
            
            if strcmp(stored_name, image_name)
                box_idx = j;
                break;
            end
        end
        
        if box_idx > 0
            % Extract the 3D fiducial mask for this image
            if iscell(box_data.fiducial_masks)
                fiducial_mask = box_data.fiducial_masks{{box_idx}};
            else
                % Handle 4D array format [num_images x depth x height x width]
                fiducial_mask = squeeze(box_data.fiducial_masks(box_idx, :, :, :));
            end
            
            fprintf('  Adding 1 fiducial mask as slave to %s\\n', image_name);
            fprintf('  Fiducial mask size: [%s]\\n', num2str(size(fiducial_mask)));
            fprintf('  Parent image size: [%s]\\n', num2str(size(current_image.data)));
            
            % Create fiducial slaves with EXACT structure from existing files
            % Ensure fiducial_mask has proper 3D dimensions
            if size(fiducial_mask, 1) == 1 && size(fiducial_mask, 2) == 1
                % Data was flattened, need to reconstruct based on parent image
                parent_size = size(current_image.data);
                fprintf('  WARNING: Fiducial mask was flattened, reconstructing to [%s]\\n', num2str(parent_size));
                % Create new mask with same dimensions as parent image
                reconstructed_mask = false(parent_size);
                % Add our boxes at the standard positions
                depth = parent_size(1); height = parent_size(2); width = parent_size(3);
                box_size = 4;  % Use the same box size
                positions = [floor(depth/3), floor(height/3), floor(width/3); ...
                           floor(2*depth/3), floor(2*height/3), floor(2*width/3)];
                for pos_idx = 1:size(positions, 1)
                    z = positions(pos_idx, 1); y = positions(pos_idx, 2); x = positions(pos_idx, 3);
                    z_start = max(1, z - floor(box_size/2));
                    z_end = min(depth, z + floor(box_size/2));
                    y_start = max(1, y - floor(box_size/2));
                    y_end = min(height, y + floor(box_size/2));
                    x_start = max(1, x - floor(box_size/2));
                    x_end = min(width, x + floor(box_size/2));
                    reconstructed_mask(z_start:z_end, y_start:y_end, x_start:x_end) = true;
                end
                fiducial_mask = reconstructed_mask;
            end
            
            % Create single fiducial slave containing ALL detected fiducials
            fiducial_slave = struct();
            fiducial_slave.Name = 'AI Fiducials';
            fiducial_slave.ImageType = '3DMASK';
            fiducial_slave.data = logical(fiducial_mask);  % Convert to logical
            fiducial_slave.FileName = '';
            fiducial_slave.Selected = 0;
            fiducial_slave.Visible = 0;
            fiducial_slave.isLoaded = 0;
            fiducial_slave.isStore = 1;
            fiducial_slave.A = eye(4);
            fiducial_slave.box = size(fiducial_mask);
            
            % Copy Anative from parent
            if isfield(current_image, 'Anative')
                fiducial_slave.Anative = current_image.Anative;
            else
                fiducial_slave.Anative = eye(4);
            end
            
            % Add single slave to the image
            if isfield(current_image, 'slaves') && ~isempty(current_image.slaves)
                % Append to existing slaves
                current_image.slaves{{end+1}} = fiducial_slave;
            else
                % Create new slaves cell array with single slave
                current_image.slaves = {{fiducial_slave}};
            end
            
            % Update the project
            project_data.images{{i}} = current_image;
            
            total_added = total_added + 1;
            fprintf('  SUCCESS: Added 1 fiducial mask to %s\\n', image_name);
        else
            fprintf('  No fiducial data found for %s\\n', image_name);
        end
    end
    
    % Save the enhanced project
    output_file = '{output_project_name}';
    fprintf('\\nSaving enhanced project: %s\\n', output_file);
    
    % Make sure we have the correct file_type
    if ~isfield(project_data, 'file_type')
        project_data.file_type = 'Reg_v2.0';
    end
    
    save(output_file, '-struct', 'project_data');
    
    fprintf('SUCCESS: Manual fiducial integration completed\\n');
    fprintf('  Total fiducials added: %d\\n', total_added);
    fprintf('  Output file: %s\\n', output_file);
    
    % Verify the saved file
    fprintf('\\nVerifying saved file...\\n');
    verify_data = load(output_file);
    fprintf('Verification:');
    fprintf('  File type: %s\\n', verify_data.file_type);
    fprintf('  Images: %d\\n', length(verify_data.images));
    
    images_with_fiducials = 0;
    for i = 1:length(verify_data.images)
        img = verify_data.images{{i}};
        if isfield(img, 'slaves') && ~isempty(img.slaves)
            images_with_fiducials = images_with_fiducials + 1;
            fprintf('  %s: %d slaves\\n', img.Name, length(img.slaves));
        end
    end
    
    fprintf('\\nSUMMARY: %d images now have fiducial slaves\\n', images_with_fiducials);
    
catch ME
    fprintf('ERROR: %s\\n', ME.message);
    fprintf('Stack trace:\\n');
    for k = 1:length(ME.stack)
        fprintf('  %s (line %d)\\n', ME.stack(k).name, ME.stack(k).line);
    end
end

fprintf('\\nIntegration completed.\\n');
end
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(matlab_script)
    
    print(f"📝 ArbuzGUI integration script created: {script_path}")
    return script_path

def run_matlab_script(script_path):
    """
    Execute the MATLAB script that uses ArbuzGUI functions
    """
    print(f"🔧 Running MATLAB script: {os.path.basename(script_path)}")
    
    script_dir = os.path.abspath(os.path.dirname(script_path))
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    
    matlab_cmd = f'matlab -batch "cd(\'{script_dir.replace(chr(92), "/")}\'); {script_name}(); exit;"'
    
    print(f"📁 Working directory: {script_dir}")
    
    try:
        result = subprocess.run(matlab_cmd, shell=True, capture_output=True, text=True, 
                              cwd=script_dir, timeout=300)
        
        print("📊 MATLAB Output:")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("⚠️ MATLAB Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ MATLAB script completed successfully")
        else:
            print(f"❌ MATLAB script failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ MATLAB script timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running MATLAB script: {e}")

def cleanup_intermediate_files(output_dir, data_file_path, script_path):
    """
    Clean up intermediate files, keeping only the final enhanced project file
    """
    print(f"🧹 Cleaning up intermediate files...")
    
    files_to_remove = []
    
    # Add intermediate data file
    if os.path.exists(data_file_path):
        files_to_remove.append(data_file_path)
    
    # Add MATLAB script
    if os.path.exists(script_path):
        files_to_remove.append(script_path)
    
    # Remove the files
    removed_count = 0
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            removed_count += 1
            print(f"   🗑️ Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"   ⚠️ Could not remove {os.path.basename(file_path)}: {e}")
    
    if removed_count > 0:
        print(f"✅ Cleaned up {removed_count} intermediate file(s)")
    
    # Show what remains
    remaining_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    if remaining_files:
        print(f"💎 Final output file(s):")
        for file in remaining_files:
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"   📄 {file} ({file_size:,} bytes)")
    else:
        print(f"⚠️ No files remain in output folder")

def process_project_with_boxes(project_file, output_dir, model=None, device=None, threshold=0.5, num_boxes=2, box_size=3):
    """
    Process all images in a project and add AI-detected or simple box fiducials
    """
    print(f"🔍 Processing project: {os.path.basename(project_file)}")
    if model is not None:
        print(f"   🤖 AI detection: threshold {threshold}")
    else:
        print(f"   📦 Box fiducials: {num_boxes} per image, size {box_size}x{box_size}x{box_size}")
    
    try:
        # Load the project file
        data = sio.loadmat(project_file, struct_as_record=False, squeeze_me=True)
        
        if 'images' not in data:
            print("❌ No 'images' field found in the project file")
            return None
        
        image_entries = data['images']
        results = []
        
        print(f"📊 Found {len(image_entries)} images to process")
        
        for entry_idx, entry in enumerate(image_entries):
            if not hasattr(entry, 'Name') or not hasattr(entry, 'data'):
                continue
                
            entry_name = str(entry.Name)
            image_data = entry.data
            
            # Determine if this is MRI or EPR and set appropriate dimensions
            is_mri = 'MRI' in entry_name.upper()
            is_epr = not is_mri
            
            # Check if image has valid loaded data
            has_valid_data = (hasattr(image_data, 'shape') and 
                            len(image_data.shape) == 3 and 
                            all(dim > 0 for dim in image_data.shape))
            
            if has_valid_data:
                # Use actual image dimensions
                image_shape = image_data.shape
                print(f"  📊 Processing {entry_name} (shape: {image_shape})")
                
                # Create AI-detected or simple box fiducials for this image
                fiducial_mask, fiducial_count = create_simple_box_fiducials(
                    image_shape, image_data, model, device, threshold, num_boxes, box_size
                )
                
            elif is_epr:
                # EPR images - use standard EPR dimensions (common for pEPRI)
                image_shape = (64, 64, 64)  # Standard EPR dimensions
                print(f"  📊 Processing {entry_name} (EPR - using standard shape: {image_shape})")
                
                # Create simple box fiducials (can't use AI without actual data)
                fiducial_mask, fiducial_count = create_simple_box_fiducials(
                    image_shape, None, None, None, threshold, num_boxes, box_size
                )
                
            elif is_mri:
                # MRI images - use standard MRI dimensions  
                image_shape = (350, 350, 20)  # Standard MRI dimensions
                print(f"  📊 Processing {entry_name} (MRI - using standard shape: {image_shape})")
                
                # Create simple box fiducials (can't use AI without actual data)
                fiducial_mask, fiducial_count = create_simple_box_fiducials(
                    image_shape, None, None, None, threshold, num_boxes, box_size
                )
            else:
                print(f"  ⚠️ Skipping {entry_name}: Cannot determine appropriate dimensions")
                continue
            
            detection_type = "AI" if (model is not None and has_valid_data) else "box"
            
            result = {
                'entry_name': entry_name,
                'image_shape': image_shape,
                'fiducial_mask': fiducial_mask,
                'num_fiducials': fiducial_count,
                'box_size': box_size,
                'image_type': 'MRI' if is_mri else 'EPR',
                'detection_method': detection_type
            }
            results.append(result)
            
            print(f"    ✅ Created {fiducial_count} {detection_type}-detected fiducials for {entry_name} ({'MRI' if is_mri else 'EPR'})")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing project file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Add AI-detected or simple box fiducials to ArbuzGUI project')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input_file', type=str,
                           help='Path to single ArbuzGUI project .mat file')
    input_group.add_argument('--input_dir', type=str,
                           help='Directory containing .mat files to process')
    
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='../models/fiducial_model.pth',
                        help='Path to trained AI model (optional)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='AI detection threshold (0-1)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_boxes', type=int, default=2,
                        help='Number of box fiducials per image (fallback)')
    parser.add_argument('--box_size', type=int, default=3,
                        help='Size of each box fiducial (fallback)')
    
    args = parser.parse_args()
    
    # Interactive prompting for missing required parameters
    if not args.input_file and not args.input_dir:
        print("\n💡 No input specified. Please provide either:")
        print("   1. Single file path")
        print("   2. Directory path")
        
        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == '1':
                args.input_file = input("Enter full path to .mat file: ").strip().strip('"')
                if not os.path.exists(args.input_file):
                    print(f"❌ Error: File '{args.input_file}' does not exist")
                    continue
                break
            elif choice == '2':
                args.input_dir = input("Enter directory path containing .mat files: ").strip().strip('"')
                if not os.path.exists(args.input_dir):
                    print(f"❌ Error: Directory '{args.input_dir}' does not exist")
                    continue
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2")
    
    if not args.output_dir:
        print("\n💾 Output directory not specified.")
        print("Default: ../data/inference")
        print("")
        response = input("Enter output directory (or press Enter for default): ").strip().strip('"')
        if response:
            args.output_dir = response
        else:
            args.output_dir = '../data/inference'
            print(f"Using default output directory: {args.output_dir}")
    
    # Handle input validation
    if args.input_file and not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return
    
    if args.input_dir and not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 Created output directory: {args.output_dir}")
    
    # Create a master timestamped folder for this batch run
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.input_file:
        # Single file: use file name
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        master_output_dir = os.path.join(args.output_dir, f"fiducials_{base_name}_{batch_timestamp}")
    else:
        # Directory: use generic batch name
        dir_name = os.path.basename(args.input_dir.rstrip('/\\'))
        master_output_dir = os.path.join(args.output_dir, f"batch_{dir_name}_{batch_timestamp}")
    
    os.makedirs(master_output_dir, exist_ok=True)
    print(f"📂 Created run folder: {os.path.basename(master_output_dir)}")
    
    # Get list of files to process
    mat_files = []
    if args.input_file:
        mat_files = [args.input_file]
    else:
        # Find all .mat files in the directory
        for file in os.listdir(args.input_dir):
            if file.lower().endswith('.mat'):
                mat_files.append(os.path.join(args.input_dir, file))
        
        if not mat_files:
            print(f"❌ No .mat files found in directory: {args.input_dir}")
            return
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting ArbuzGUI fiducial integration")
    if args.input_file:
        print(f"   Input file: {args.input_file}")
    else:
        print(f"   Input directory: {args.input_dir}")
        print(f"   Found {len(mat_files)} .mat files")
    print(f"   Master output: {master_output_dir}")
    print(f"   Device: {device}")
    
    # Try to load AI model
    model = None
    if os.path.exists(args.model_path):
        try:
            print(f"   🤖 Loading AI model: {args.model_path}")
            model = load_model(args.model_path, device)
            print(f"   🎯 AI threshold: {args.threshold}")
        except Exception as e:
            print(f"   ⚠️  AI model loading failed: {e}")
            print(f"   📦 Will use static boxes instead")
            model = None
    else:
        print(f"   ⚠️  AI model not found: {args.model_path}")
        print(f"   📦 Using static box creation")
    
    if model is None:
        print(f"   Boxes per image: {args.num_boxes}")
        print(f"   Box size: {args.box_size}x{args.box_size}x{args.box_size}")
    
    # Process all files
    all_results = []
    total_processed = 0
    
    for mat_file in mat_files:
        print(f"\n📂 Processing: {os.path.basename(mat_file)}")
        project_name = os.path.splitext(os.path.basename(mat_file))[0]
        
        try:
            results = process_project_with_boxes(
                mat_file, master_output_dir, model, device, args.threshold, args.num_boxes, args.box_size
            )
            
            if results:
                # Save fiducial data for ArbuzGUI (this creates its own timestamped subfolder)
                data_file_path, unique_output_dir = save_box_data_for_arbuz(
                    results, master_output_dir, project_name
                )
                
                # Create ArbuzGUI integration script
                script_path = create_arbuz_integration_script(
                    data_file_path, mat_file, unique_output_dir, 'add_fiducials_manual'
                )
                
                # Run the MATLAB script that uses ArbuzGUI functions
                run_matlab_script(script_path)
                
                # Clean up intermediate files, keeping only the final enhanced project file
                cleanup_intermediate_files(unique_output_dir, data_file_path, script_path)
                
                all_results.extend(results)
                total_processed += 1
                print(f"✅ Successfully processed {os.path.basename(mat_file)}")
                print(f"   📁 Results saved to: {os.path.basename(unique_output_dir)}")
            else:
                print(f"⚠️ No results for {os.path.basename(mat_file)}")
                
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(mat_file)}: {e}")
            continue
    
    if not all_results:
        print("❌ No files were successfully processed")
        return
    
    # Overall Summary
    total_fiducials = sum(r['num_fiducials'] for r in all_results)
    mri_count = len([r for r in all_results if r['image_type'] == 'MRI'])
    epr_count = len([r for r in all_results if r['image_type'] == 'EPR'])
    
    ai_count = len([r for r in all_results if r['detection_method'] == 'AI'])
    box_count = len([r for r in all_results if r['detection_method'] == 'box'])
    
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   Files processed: {total_processed}/{len(mat_files)}")
    print(f"   Total images: {len(all_results)} ({mri_count} MRI, {epr_count} EPR)")
    print(f"   Total fiducials: {total_fiducials}")
    if model is not None:
        print(f"   AI detections: {ai_count} images")
        print(f"   Box fallbacks: {box_count} images")
    else:
        print(f"   Static boxes: {len(all_results)} images")
    print(f"   Output directory: {master_output_dir}")
    
    print(f"\n📄 MRI Images:")
    for result in all_results:
        if result['image_type'] == 'MRI':
            method_emoji = "🤖" if result['detection_method'] == 'AI' else "📦"
            print(f"   🏥 {result['entry_name']}: {result['num_fiducials']} fiducials {method_emoji} ({result['detection_method']}, shape: {result['image_shape']})")
    
    print(f"\n📄 EPR Images:")
    for result in all_results:
        if result['image_type'] == 'EPR':
            method_emoji = "🤖" if result['detection_method'] == 'AI' else "📦"
            print(f"   🔬 {result['entry_name']}: {result['num_fiducials']} fiducials {method_emoji} ({result['detection_method']}, shape: {result['image_shape']})")
    
    detection_method = "AI + fallback boxes" if model is not None else "static boxes"
    print(f"\n✅ Batch fiducial integration completed using {detection_method}!")
    print(f"📂 All results saved to: {master_output_dir}")

if __name__ == "__main__":
    main()
