import torch
import numpy as np
import scipy.io as sio
import os
from scipy import ndimage
import argparse
from datetime import datetime

# Import the model class (clean version without training code)
from fiducial_model import FiducialUNet3D

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

def load_model(model_path, device):
    """Load the trained fiducial detection model"""
    model = FiducialUNet3D()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from {model_path}")
            if 'best_val_loss' in checkpoint:
                print(f"   Best validation loss: {checkpoint['best_val_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.to(device)
    model.eval()
    return model

def predict_fiducials(model, image_data, device, target_size=(64, 64, 64), threshold=0.5):
    """
    Predict fiducials in a 3D image (entire volume processing)
    
    Args:
        model: Trained fiducial detection model
        image_data: 3D numpy array (entire volume)
        device: torch device
        target_size: Target size for model input (will resize if needed)
        threshold: Threshold for binary prediction
    
    Returns:
        predicted_mask: Binary mask of predicted fiducials (same size as input)
        confidence_map: Confidence map (0-1, same size as input)
    """
    original_shape = image_data.shape
    
    print(f"🔍 Processing entire 3D volume: {original_shape}")
    
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
    binary_prediction = (confidence_map > threshold).astype(np.float64)
    
    # Resize back to original size if needed
    if original_shape != target_size:
        confidence_map_resized = resize_3d_volume(confidence_map, original_shape)
        binary_prediction_resized = resize_3d_volume(binary_prediction.astype(np.float32), original_shape)
        binary_prediction_resized = (binary_prediction_resized > 0.5).astype(np.float64)
    else:
        confidence_map_resized = confidence_map
        binary_prediction_resized = binary_prediction
    
    # Count detected fiducials (connected components)
    labeled_mask, num_components = ndimage.label(binary_prediction_resized)
    print(f"✅ Detected {num_components} fiducial regions in entire volume")
    print(f"📏 Total fiducial volume: {binary_prediction_resized.sum()} voxels")
    
    return binary_prediction_resized, confidence_map_resized

def create_mat_output_with_fiducials(original_mat_data, results, output_path):
    """
    Create a new .mat file with the same structure as input but with added fiducial masks
    
    Args:
        original_mat_data: Original loaded .mat file data
        results: List of prediction results
        output_path: Path to save the new .mat file
    """
    print(f"💾 Creating enhanced .mat file with fiducial masks...")
    
    # Create a copy of the original data structure EXACTLY (no new keys!)
    output_data = {}
    
    # Copy all original fields EXACTLY
    for key, value in original_mat_data.items():
        if not key.startswith('__'):  # Skip MATLAB metadata
            output_data[key] = value
    
    # Enhance the images structure with fiducial masks
    if 'images' in original_mat_data:
        image_entries = original_mat_data['images']
        
        # Create enhanced image entries
        enhanced_entries = []
        
        for entry_idx, entry in enumerate(image_entries):
            if hasattr(entry, 'Name') and hasattr(entry, 'data'):
                entry_name = str(entry.Name)
                
                # Find corresponding result
                matching_result = None
                for result in results:
                    if result['entry_name'] == entry_name:
                        matching_result = result
                        break
                
                if matching_result:
                    # Create enhanced entry with fiducial data in slaves (EXACT structure match)
                    enhanced_entry = type('obj', (object,), {})()
                    
                    # Copy all original fields with EXACT ORIGINAL DATA TYPES
                    for attr_name in dir(entry):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(entry, attr_name)
                                # Keep EXACT original data types (no conversion!)
                                setattr(enhanced_entry, attr_name, attr_value)
                            except:
                                pass
                    
                    # Keep original data types (DO NOT convert to double)
                    # MRI: uint16, EPR: float64, as per examples
                    
                    # Create fiducial slave entry if fiducials were detected
                    if matching_result['num_fiducials'] > 0:
                        fiducial_slave = type('obj', (object,), {})()
                        
                        # Set slave properties EXACTLY like examples - ALL VALUES MUST BE DOUBLE
                        fiducial_slave.Name = 'Fiducials_AI'
                        fiducial_slave.data = matching_result['predicted_mask'].astype(np.float64)  # MUST be double for ArbuzGUI
                        fiducial_slave.FileName = ''
                        fiducial_slave.ImageType = 'MASK'
                        fiducial_slave.Selected = 0  # Keep as int like original
                        fiducial_slave.Visible = 1   # Keep as int like original
                        fiducial_slave.isLoaded = 1  # Keep as int like original
                        fiducial_slave.isStore = 1   # Keep as int like original
                        
                        # Copy geometric properties from parent with EXACT ORIGINAL DATA TYPES
                        if hasattr(entry, 'A'):
                            fiducial_slave.A = entry.A  # Keep original dtype (uint8)
                        if hasattr(entry, 'Anative'):
                            fiducial_slave.Anative = entry.Anative  # Keep original dtype (float64)
                        if hasattr(entry, 'Aprime'):
                            fiducial_slave.Aprime = entry.Aprime  # Keep original dtype (uint8)
                        if hasattr(entry, 'Anext'):
                            fiducial_slave.Anext = entry.Anext  # Keep original dtype (uint8)
                        if hasattr(entry, 'box'):
                            fiducial_slave.box = entry.box  # Keep original dtype (uint16 or uint8)
                        
                        # Add slave to enhanced entry as numpy array
                        enhanced_entry.slaves = np.array([fiducial_slave], dtype=object)
                    # If no fiducials, slaves stays as original empty array
                    
                    enhanced_entries.append(enhanced_entry)
                    
                    if matching_result['num_fiducials'] > 0:
                        print(f"  ✅ Enhanced {entry_name} with {matching_result['num_fiducials']} fiducials (added to slaves)")
                    else:
                        print(f"  ✅ Enhanced {entry_name} with 0 fiducials (no slaves added)")
                else:
                    # Keep original entry but convert image data to double
                    enhanced_entry = type('obj', (object,), {})()
                    
                    # Copy all original fields with EXACT ORIGINAL DATA TYPES 
                    for attr_name in dir(entry):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(entry, attr_name)
                                # Keep EXACT original data types (no conversion!)
                                setattr(enhanced_entry, attr_name, attr_value)
                            except:
                                pass
                    
                    enhanced_entries.append(enhanced_entry)
                    print(f"  ⚠️ No fiducial data for {entry_name}, keeping original (exact data types preserved)")
            else:
                # Keep entries that don't have the expected structure
                enhanced_entries.append(entry)
        
        # Update the images field
        output_data['images'] = enhanced_entries
    
    # Save the enhanced .mat file (NO EXTRA METADATA!)
    try:
        sio.savemat(output_path, output_data, format='5', long_field_names=True)
        print(f"✅ Enhanced .mat file saved to: {output_path}")
        
        # Print summary of what was added
        print(f"📊 Enhancement Summary:")
        print(f"   - Original images: {len(image_entries) if 'images' in original_mat_data else 0}")
        print(f"   - Enhanced with fiducials: {len([r for r in results if r['num_fiducials'] > 0])}")
        print(f"   - Total fiducials detected: {sum(r['num_fiducials'] for r in results)}")
        print(f"📝 Structure preserved EXACTLY like original:")
        print(f"   - Original data types maintained (uint8, uint16, float64)")
        print(f"   - Fiducial masks as float64 slaves (0-1 range)")
        print(f"   - No extra keys added")
        
    except Exception as e:
        print(f"❌ Error saving enhanced .mat file: {e}")
        import traceback
        traceback.print_exc()

def analyze_fiducials(predicted_mask, confidence_map, min_volume=5):
    """
    Analyze detected fiducials and return detailed information
    
    Args:
        predicted_mask: Binary mask of fiducials
        confidence_map: Confidence values
        min_volume: Minimum volume for valid fiducials
    
    Returns:
        fiducial_info: List of fiducial properties
    """
    # Label connected components
    labeled_mask, num_regions = ndimage.label(predicted_mask)
    
    fiducial_info = []
    valid_count = 0
    
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled_mask == region_id)
        volume = region_mask.sum()
        
        if volume >= min_volume:
            valid_count += 1
            
            # Calculate centroid
            indices = np.where(region_mask)
            centroid = (
                np.mean(indices[0]),
                np.mean(indices[1]),
                np.mean(indices[2])
            )
            
            # Calculate confidence statistics
            region_confidences = confidence_map[region_mask]
            avg_conf = np.mean(region_confidences)
            max_conf = np.max(region_confidences)
            
            fiducial_info.append({
                'id': valid_count,
                'volume': volume,
                'centroid': centroid,
                'avg_confidence': avg_conf,
                'max_confidence': max_conf
            })
    
    return fiducial_info

def predict_on_mat_file(mat_file_path, model, device, output_dir=None, threshold=0.5):
    """
    Predict fiducials on all images in a .mat file
    Creates an enhanced .mat file with same structure + fiducial masks
    
    Args:
        mat_file_path: Path to .mat file
        model: Trained model
        device: torch device
        output_dir: Directory to save results (optional)
        threshold: Prediction threshold
    """
    print(f"\n🔍 Processing: {os.path.basename(mat_file_path)}")
    
    try:
        # Load the .mat file
        data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
        
        if 'images' not in data:
            print("❌ No 'images' field found in the file")
            return None
        
        image_entries = data['images']
        results = []
        
        for entry_idx, entry in enumerate(image_entries):
            if not hasattr(entry, 'Name') or not hasattr(entry, 'data'):
                continue
                
            entry_name = str(entry.Name)
            image_data = entry.data
            
            if not hasattr(image_data, 'shape') or len(image_data.shape) != 3:
                print(f"  ⚠️ Skipping {entry_name}: Invalid image data")
                continue
            
            print(f"  📊 Processing {entry_name} (shape: {image_data.shape})")
            
            # Predict fiducials on entire 3D volume
            predicted_mask, confidence_map = predict_fiducials(
                model, image_data, device, threshold=threshold
            )
            
            # Analyze fiducials
            fiducial_info = analyze_fiducials(predicted_mask, confidence_map, min_volume=5)
            
            print(f"    ✅ Detected {len(fiducial_info)} valid fiducials")
            for i, fid in enumerate(fiducial_info):
                print(f"       Fiducial {i+1}: Volume={fid['volume']}, Centroid=({fid['centroid'][0]:.1f}, {fid['centroid'][1]:.1f}, {fid['centroid'][2]:.1f})")
            
            result = {
                'entry_name': entry_name,
                'original_shape': image_data.shape,
                'predicted_mask': predicted_mask,
                'confidence_map': confidence_map,
                'fiducial_info': fiducial_info,
                'num_fiducials': len(fiducial_info),
                'total_volume': predicted_mask.sum()
            }
            results.append(result)
        
        # Save enhanced .mat file with fiducials
        if output_dir and results:
            # Create unique timestamped folder to avoid overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
            unique_output_dir = os.path.join(output_dir, f"{base_name}_{timestamp}")
            os.makedirs(unique_output_dir, exist_ok=True)
            
            # Create enhanced .mat file in the unique folder
            enhanced_mat_path = os.path.join(unique_output_dir, f"{base_name}_with_fiducials.mat")
            
            create_mat_output_with_fiducials(data, results, enhanced_mat_path)
            
            print(f"📁 Results saved to unique folder: {unique_output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing {mat_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Predict fiducials in 3D medical images')
    parser.add_argument('--model_path', type=str, default='../models/fiducial_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--input_file', type=str, 
                        help='Path to a specific .mat file to process')
    parser.add_argument('--input_dir', type=str, default='../data/training',
                        help='Directory containing .mat files to process')
    parser.add_argument('--output_dir', type=str, default='../data/inference',
                        help='Directory to save prediction results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold (0-1)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.model_path, device)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please make sure you have trained the model first using train_fiducial_model.py")
        return
    
    # Process files
    if args.input_file:
        # Process single file
        if os.path.exists(args.input_file):
            results = predict_on_mat_file(args.input_file, model, device, 
                                        args.output_dir, args.threshold)
        else:
            print(f"❌ File not found: {args.input_file}")
    else:
        # Process all files in directory
        if not os.path.exists(args.input_dir):
            print(f"❌ Directory not found: {args.input_dir}")
            return
        
        mat_files = [f for f in os.listdir(args.input_dir) if f.endswith('.mat')]
        
        if not mat_files:
            print(f"❌ No .mat files found in {args.input_dir}")
            return
        
        print(f"📁 Found {len(mat_files)} .mat files")
        
        all_results = []
        for mat_file in mat_files:
            mat_path = os.path.join(args.input_dir, mat_file)
            # Each file gets its own timestamped folder
            results = predict_on_mat_file(mat_path, model, device, 
                                        args.output_dir, args.threshold)
            if results:
                all_results.extend(results)
        
        # Summary statistics
        if all_results:
            total_fiducials = sum(r['num_fiducials'] for r in all_results)
            total_volume = sum(r['total_volume'] for r in all_results)
            
            print(f"\\n📊 Summary:")
            print(f"   Processed {len(all_results)} images")
            print(f"   Total detected fiducials: {total_fiducials}")
            print(f"   Total fiducial volume: {total_volume} voxels")
            print(f"   Average fiducials per image: {total_fiducials/len(all_results):.1f}")
            
            # Show detailed breakdown
            for result in all_results:
                print(f"   📄 {result['entry_name']}: {result['num_fiducials']} fiducials, {result['total_volume']} voxels")
    
    print(f"\\n✅ Prediction completed!")
    if args.output_dir:
        print(f"   Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
