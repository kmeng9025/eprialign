"""
Final kidney segmentation pipeline using trained AI model
Replaces static boxes with intelligent kidney detection
"""
import numpy as np
import scipy.io
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def load_kidney_model():
    """Load the trained kidney segmentation model"""
    # Find the latest model file
    model_files = list(Path(".").glob("kidney_model_*.joblib"))
    
    if not model_files:
        raise FileNotFoundError("No trained kidney model found! Run train_final_model.py first.")
    
    # Get the latest model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"🧠 Loading model: {latest_model}")
    model_data = joblib.load(latest_model)
    
    info = model_data['training_info']
    print(f"   Training accuracy: {info['train_acc']:.3f}")
    print(f"   Validation accuracy: {info['val_acc']:.3f}")
    print(f"   F1 Score: {info['f1_score']:.3f}")
    
    return model_data

def predict_kidney_mask(mri_volume, model_data):
    """Predict kidney mask for a given MRI volume"""
    model = model_data['model']
    
    h, w, d = mri_volume.shape
    prediction = np.zeros((h, w, d), dtype=np.uint8)
    
    print(f"🔮 Predicting kidney mask for {h}×{w}×{d} volume...")
    
    # Process slice by slice to manage memory
    for z in range(d):
        slice_features = []
        coords = []
        
        for y in range(h):
            for x in range(w):
                intensity = mri_volume[y, x, z]
                pos_x = x / w
                pos_y = y / h
                pos_z = z / d
                
                center_x, center_y, center_z = w/2, h/2, d/2
                dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2 + center_z**2)
                dist_center_norm = dist_center / max_dist
                
                local_mean = intensity
                if 1 <= x < w-1 and 1 <= y < h-1 and 1 <= z < d-1:
                    local_mean = np.mean(mri_volume[y-1:y+2, x-1:x+2, z-1:z+2])
                
                features = [intensity, pos_x, pos_y, pos_z, dist_center_norm, local_mean]
                slice_features.append(features)
                coords.append((y, x))
        
        # Predict slice
        slice_features = np.array(slice_features)
        slice_predictions = model.predict(slice_features)
        
        # Fill prediction mask
        for i, (y, x) in enumerate(coords):
            prediction[y, x, z] = slice_predictions[i]
        
        if (z + 1) % 5 == 0 or z == d - 1:
            print(f"  Processed {z+1}/{d} slices")
    
    kidney_pixels = np.sum(prediction)
    total_pixels = prediction.size
    coverage = (kidney_pixels / total_pixels) * 100
    
    print(f"✅ Prediction complete: {kidney_pixels} kidney pixels ({coverage:.2f}%)")
    
    return prediction

def extract_kidney_bounding_boxes(kidney_mask, min_size=100):
    """Extract bounding boxes for detected kidneys"""
    
    # Find connected components (simple approach)
    labeled_mask = np.zeros_like(kidney_mask)
    
    # Simple connected component labeling
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(kidney_mask)
    
    print(f"🔍 Found {num_features} connected components")
    
    boxes = []
    for label in range(1, num_features + 1):
        component = (labeled_array == label)
        component_size = np.sum(component)
        
        if component_size < min_size:
            continue
        
        # Find bounding box
        coords = np.where(component)
        min_y, max_y = np.min(coords[0]), np.max(coords[0])
        min_x, max_x = np.min(coords[1]), np.max(coords[1])
        min_z, max_z = np.min(coords[2]), np.max(coords[2])
        
        # Add some padding
        h, w, d = kidney_mask.shape
        padding = 5
        
        min_y = max(0, min_y - padding)
        max_y = min(h - 1, max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(w - 1, max_x + padding)
        min_z = max(0, min_z - padding)
        max_z = min(d - 1, max_z + padding)
        
        box = {
            'size': component_size,
            'bounds': (min_y, max_y, min_x, max_x, min_z, max_z),
            'center': ((min_y + max_y) // 2, (min_x + max_x) // 2, (min_z + max_z) // 2)
        }
        
        boxes.append(box)
        print(f"  Kidney {label}: {component_size} voxels, center {box['center']}")
    
    # Sort by size (largest first)
    boxes.sort(key=lambda x: x['size'], reverse=True)
    
    return boxes

def process_mri_file(mat_file, model_data):
    """Process a single MRI file with AI kidney detection"""
    print(f"\n🏥 PROCESSING: {mat_file}")
    print("="*60)
    
    try:
        # Load the .mat file
        data = scipy.io.loadmat(mat_file)
        
        # Find MRI data
        mri_data = None
        data_key = None
        
        for key, value in data.items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, 'shape') and len(value.shape) == 3:
                if value.shape[0] == 350 and value.shape[1] == 350:  # MRI dimensions
                    mri_data = value
                    data_key = key
                    break
        
        if mri_data is None:
            print("❌ No MRI data found in file")
            return None
        
        print(f"📊 Found MRI data: {data_key} {mri_data.shape}")
        print(f"   Value range: [{np.min(mri_data):.3f}, {np.max(mri_data):.3f}]")
        
        # Normalize MRI data
        mri_normalized = mri_data.astype(np.float32)
        if np.max(mri_normalized) > 1.0:
            mri_normalized = mri_normalized / np.max(mri_normalized)
        
        # Predict kidney mask
        kidney_mask = predict_kidney_mask(mri_normalized, model_data)
        
        # Extract bounding boxes
        boxes = extract_kidney_bounding_boxes(kidney_mask)
        
        if len(boxes) == 0:
            print("⚠️  No kidneys detected!")
            return None
        
        print(f"🎯 Detected {len(boxes)} kidney(s):")
        for i, box in enumerate(boxes):
            bounds = box['bounds']
            size = box['size']
            coverage = (size / kidney_mask.size) * 100
            print(f"   Kidney {i+1}: bounds {bounds}, {size} voxels ({coverage:.2f}%)")
        
        # Create visualization
        create_kidney_visualization(mri_normalized, kidney_mask, boxes, mat_file)
        
        result = {
            'file': mat_file,
            'mri_shape': mri_data.shape,
            'kidney_mask': kidney_mask,
            'bounding_boxes': boxes,
            'total_kidney_voxels': np.sum(kidney_mask),
            'coverage_percent': (np.sum(kidney_mask) / kidney_mask.size) * 100
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing {mat_file}: {e}")
        return None

def create_kidney_visualization(mri_data, kidney_mask, boxes, filename):
    """Create visualization of kidney detection results"""
    
    # Choose middle slice for visualization
    h, w, d = mri_data.shape
    mid_slice = d // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original MRI
    axes[0].imshow(mri_data[:, :, mid_slice], cmap='gray')
    axes[0].set_title(f'Original MRI (slice {mid_slice})')
    axes[0].axis('off')
    
    # Kidney mask overlay
    axes[1].imshow(mri_data[:, :, mid_slice], cmap='gray')
    mask_overlay = np.ma.masked_where(kidney_mask[:, :, mid_slice] == 0, kidney_mask[:, :, mid_slice])
    axes[1].imshow(mask_overlay, alpha=0.5, cmap='Reds')
    axes[1].set_title('AI Kidney Detection')
    axes[1].axis('off')
    
    # Bounding boxes
    axes[2].imshow(mri_data[:, :, mid_slice], cmap='gray')
    
    # Draw bounding boxes that intersect this slice
    for i, box in enumerate(boxes):
        min_y, max_y, min_x, max_x, min_z, max_z = box['bounds']
        
        if min_z <= mid_slice <= max_z:
            # Draw rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[2].add_patch(rect)
            
            # Add label
            axes[2].text(min_x, min_y - 5, f'Kidney {i+1}', 
                        color='red', fontsize=10, weight='bold')
    
    axes[2].set_title('Bounding Boxes')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_name = f"kidney_detection_{Path(filename).stem}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Visualization saved: {output_name}")

def main():
    """Main kidney segmentation pipeline"""
    print("🚀 AI KIDNEY SEGMENTATION PIPELINE")
    print("="*50)
    
    # Load model
    try:
        model_data = load_kidney_model()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Find test files
    test_files = []
    
    # Look for .mat files in data directory
    data_dir = Path("../../data")
    if data_dir.exists():
        test_files.extend(data_dir.glob("*.mat"))
    
    # Look in current directory
    test_files.extend(Path(".").glob("*.mat"))
    
    if not test_files:
        print("❌ No .mat files found for testing!")
        print("   Place .mat files in current directory or ../../data/")
        return
    
    print(f"📁 Found {len(test_files)} test files")
    
    # Process each file
    results = []
    
    for mat_file in test_files[:3]:  # Process first 3 files
        result = process_mri_file(str(mat_file), model_data)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n📋 PROCESSING SUMMARY")
    print("="*30)
    print(f"   Files processed: {len(results)}")
    
    for result in results:
        print(f"   {Path(result['file']).name}:")
        print(f"     Shape: {result['mri_shape']}")
        print(f"     Kidneys: {len(result['bounding_boxes'])}")
        print(f"     Coverage: {result['coverage_percent']:.2f}%")
    
    print(f"\n🎉 AI kidney segmentation complete!")
    print(f"   Model successfully replaced static boxes!")

if __name__ == "__main__":
    main()
