#!/usr/bin/env python3
"""
Check training data to see if kidney masks were too large or incorrect
"""

import numpy as np
import pickle
import os

def check_training_data():
    """Check the actual training data that was used"""
    
    print("🔍 CHECKING TRAINING DATA QUALITY")
    print("=" * 60)
    
    training_file = "kidney_training_data.pkl"
    if not os.path.exists(training_file):
        print(f"❌ Training data file not found: {training_file}")
        return
    
    with open(training_file, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"📊 Training data structure:")
    for key, value in training_data.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape {value.shape}, type {value.dtype}")
        elif isinstance(value, list):
            print(f"   {key}: list of {len(value)} items")
            if len(value) > 0:
                first_item = value[0]
                if hasattr(first_item, 'shape'):
                    print(f"      First item shape: {first_item.shape}, type: {first_item.dtype}")
        else:
            print(f"   {key}: {type(value)}")
    
    # Get the kidney masks
    kidney_masks = training_data['kidney_masks']
    mri_volumes = training_data['mri_volumes']
    
    print(f"\n🎯 ANALYZING {len(kidney_masks)} TRAINING SAMPLES:")
    
    total_coverage = 0
    problematic_samples = []
    
    for i in range(len(kidney_masks)):
        mask = kidney_masks[i]
        mri = mri_volumes[i]
        
        kidney_voxels = np.count_nonzero(mask)
        total_voxels = np.prod(mask.shape)
        coverage = kidney_voxels / total_voxels * 100
        total_coverage += coverage
        
        unique_vals = np.unique(mask)
        
        print(f"\n   📂 Sample {i}:")
        print(f"      MRI shape: {mri.shape}, Mask shape: {mask.shape}")
        print(f"      Kidney voxels: {kidney_voxels:,} / {total_voxels:,}")
        print(f"      Coverage: {coverage:.2f}%")
        print(f"      Mask values: {unique_vals}")
        
        if coverage > 50:
            print(f"      🚨 PROBLEM: {coverage:.2f}% coverage is way too high for kidneys!")
            problematic_samples.append(i)
        elif coverage > 10:
            print(f"      ⚠️  WARNING: {coverage:.2f}% coverage is high for kidneys")
            problematic_samples.append(i)
        elif coverage < 0.1:
            print(f"      ⚠️  WARNING: {coverage:.2f}% coverage might be too low")
        else:
            print(f"      ✅ Normal kidney coverage ({coverage:.2f}%)")
    
    avg_coverage = total_coverage / len(kidney_masks)
    print(f"\n📊 OVERALL TRAINING DATA ANALYSIS:")
    print(f"   Average kidney coverage: {avg_coverage:.2f}%")
    print(f"   Problematic samples: {len(problematic_samples)} / {len(kidney_masks)}")
    print(f"   Problematic sample IDs: {problematic_samples}")
    
    if avg_coverage > 20:
        print(f"\n🚨 CRITICAL ISSUE: Average coverage {avg_coverage:.2f}% is way too high!")
        print(f"   Normal kidney coverage should be ~1-5% of total volume")
        print(f"   This explains why the model learned to predict everything as kidney")
        print(f"   The training data contains incorrect/oversized kidney masks")
    elif len(problematic_samples) > len(kidney_masks) / 2:
        print(f"\n⚠️  ISSUE: {len(problematic_samples)} out of {len(kidney_masks)} samples have problematic coverage")
        print(f"   This could cause the model to overpredict kidneys")
    else:
        print(f"\n✅ Training data coverage looks reasonable")
    
    return problematic_samples, avg_coverage

def recommend_fix():
    """Recommend how to fix the training data"""
    
    print(f"\n🔧 RECOMMENDED FIXES:")
    print("=" * 60)
    
    print(f"1. 🔍 Re-examine training data extraction:")
    print(f"   - Check extract_true_kidney_masks.py")
    print(f"   - Verify that 'masks' in training files are actual kidney regions")
    print(f"   - Ensure masks aren't entire body/organ regions")
    
    print(f"\n2. 📊 Re-train with corrected data:")
    print(f"   - Extract only true kidney regions (small, organ-shaped)")
    print(f"   - Target coverage should be 1-5% of total volume")
    print(f"   - Use multiple smaller kidney regions rather than large masks")
    
    print(f"\n3. 🎯 Model architecture considerations:")
    print(f"   - Current model might be too simple for this task")
    print(f"   - Consider adding more regularization")
    print(f"   - Use class weighting to handle kidney/background imbalance")
    
    print(f"\n4. 🧹 Immediate fix:")
    print(f"   - Increase threshold from 0.3 to 0.7 or higher")
    print(f"   - This will dramatically reduce false positives")
    print(f"   - But won't fix the underlying model issue")

if __name__ == "__main__":
    problematic_samples, avg_coverage = check_training_data()
    recommend_fix()
