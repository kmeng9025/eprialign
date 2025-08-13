#!/usr/bin/env python3
"""
Test script for the kidney segmentation pipeline

This script runs the kidney segmentation pipeline on the test data file.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from kidney_segmentation_pipeline import KidneySegmentationPipeline

def main():
    """Run the test"""
    
    # Define paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "../../../data/training/withoutROIwithMRI.mat"
    output_dir = script_dir / "../../../data/inference"
    
    # Make paths absolute
    input_file = input_file.resolve()
    output_dir = output_dir.resolve()
    
    print(f"Test script starting...")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Check if input file exists
    if not input_file.exists():
        print(f"ERROR: Input file does not exist: {input_file}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create and run pipeline
        pipeline = KidneySegmentationPipeline(
            input_file=str(input_file),
            output_dir=str(output_dir)
        )
        
        pipeline.run()
        
        print("Test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
