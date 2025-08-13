"""
Extract and save kidney training data as numpy arrays
"""
import numpy as np
import pickle
import os
from pathlib import Path

def extract_and_save_training_data():
    """Extract kidney training data and save as numpy arrays"""
    print("📦 EXTRACTING KIDNEY TRAINING DATA")
    print("="*50)
    
    # Load the extracted kidney samples
    if not os.path.exists("kidney_training_samples.pkl"):
        print("❌ Need to run extract_true_kidney_masks.py first!")
        return False
    
    with open("kidney_training_samples.pkl", 'rb') as f:
        samples = pickle.load(f)
    
    print(f"📋 Found {len(samples)} kidney training samples")
    
    # Prepare lists to store data
    mri_volumes = []
    kidney_masks = []
    sample_metadata = []
    
    for i, sample in enumerate(samples):
        print(f"  Processing {i+1}/{len(samples)}: {sample['file']}")
        
        # Get the data
        mri_data = sample['mri_data'].astype(np.float32)
        kidney_mask = sample['kidney_mask'].astype(np.uint8)
        
        # Normalize MRI data to [0, 1] range
        if mri_data.max() > mri_data.min():
            mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
        
        # Store the data
        mri_volumes.append(mri_data)
        kidney_masks.append(kidney_mask)
        
        # Store metadata
        metadata = {
            'file': sample['file'],
            'image_idx': sample['image_idx'],
            'slave_idx': sample['slave_idx'],
            'shape': sample['mri_shape'],
            'mask_name': sample['mask_name'],
            'coverage': sample['mask_coverage']
        }
        sample_metadata.append(metadata)
        
        print(f"    Shape: {mri_data.shape}, Kidney pixels: {np.sum(kidney_mask)} ({sample['mask_coverage']:.2f}%)")
    
    # Save training data
    print("\n💾 Saving training data...")
    
    # Save each volume separately with indices
    training_data = {
        'mri_volumes': mri_volumes,
        'kidney_masks': kidney_masks,
        'metadata': sample_metadata
    }
    
    # Save as pickle since shapes differ
    with open("kidney_training_data.pkl", 'wb') as f:
        pickle.dump(training_data, f)
    print(f"   Saved training data: kidney_training_data.pkl ({len(mri_volumes)} volumes)")
    
    # Also save as individual numpy files for easy access
    for i, (mri, mask) in enumerate(zip(mri_volumes, kidney_masks)):
        np.save(f"kidney_mri_{i:02d}.npy", mri)
        np.save(f"kidney_mask_{i:02d}.npy", mask)
    print(f"   Also saved individual files: kidney_mri_XX.npy, kidney_mask_XX.npy")
    
    # Print summary
    print(f"\n📊 TRAINING DATA SUMMARY")
    print(f"   Total training pairs: {len(mri_volumes)}")
    
    # Group by shape
    shape_counts = {}
    for metadata in sample_metadata:
        shape = metadata['shape']
        shape_str = f"{shape[0]}×{shape[1]}×{shape[2]}"
        shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
    
    print(f"   Shape distribution:")
    for shape, count in shape_counts.items():
        print(f"     {shape}: {count} volumes")
    
    # Coverage statistics
    coverages = [meta['coverage'] for meta in sample_metadata]
    print(f"   Kidney coverage:")
    print(f"     Min: {min(coverages):.2f}%")
    print(f"     Max: {max(coverages):.2f}%") 
    print(f"     Mean: {np.mean(coverages):.2f}%")
    
    print(f"\n✅ Training data extraction complete!")
    return True

def load_training_data():
    """Load the saved training data"""
    print("📂 Loading saved training data...")
    
    try:
        with open("kidney_training_data.pkl", 'rb') as f:
            training_data = pickle.load(f)
        
        mri_volumes = training_data['mri_volumes']
        kidney_masks = training_data['kidney_masks'] 
        metadata = training_data['metadata']
        
        print(f"✅ Loaded {len(mri_volumes)} training pairs")
        return mri_volumes, kidney_masks, metadata
        
    except FileNotFoundError as e:
        print(f"❌ Training data file not found: {e}")
        print("   Run this script first to extract training data")
        return None, None, None

def verify_training_data():
    """Verify the saved training data"""
    print("\n🔍 VERIFYING TRAINING DATA")
    print("="*40)
    
    mri_volumes, kidney_masks, metadata = load_training_data()
    
    if mri_volumes is None:
        return False
    
    print(f"📊 Verification results:")
    print(f"   MRI volumes: {len(mri_volumes)}")
    print(f"   Kidney masks: {len(kidney_masks)}")
    print(f"   Metadata entries: {len(metadata)}")
    
    # Check each pair
    for i in range(len(mri_volumes)):
        mri = mri_volumes[i]
        mask = kidney_masks[i]
        meta = metadata[i]
        
        print(f"   Sample {i+1}: {meta['file']}")
        print(f"     MRI shape: {mri.shape}")
        print(f"     Mask shape: {mask.shape}")
        print(f"     Shapes match: {mri.shape == mask.shape}")
        print(f"     MRI range: [{mri.min():.3f}, {mri.max():.3f}]")
        print(f"     Mask values: {np.unique(mask)}")
        print(f"     Kidney pixels: {np.sum(mask)} ({meta['coverage']:.2f}%)")
    
    print(f"\n✅ Training data verification complete!")
    return True

if __name__ == "__main__":
    # Extract and save training data
    success = extract_and_save_training_data()
    
    if success:
        # Verify the saved data
        verify_training_data()
        
        print(f"\n🎉 READY FOR MODEL TRAINING!")
        print(f"   Files created:")
        print(f"     • kidney_training_data.pkl")
        print(f"     • kidney_mri_XX.npy (individual files)")
        print(f"     • kidney_mask_XX.npy (individual files)")
        print(f"   Next step: Train model using these files")
    else:
        print(f"\n❌ Failed to extract training data")
