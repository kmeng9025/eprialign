#!/usr/bin/env python3
"""
Static Kidney Boxes - Simple kidney box addition to Arbuz projects
================================================================

This script adds two static kidney boxes (left and right) as slaves to Arbuz projects.
Much simpler than AI detection - just places boxes at typical kidney locations.

Author: AI Assistant
Date: 2025-08-13
"""

import os
import sys
import subprocess
from datetime import datetime

def add_static_kidneys(input_file, output_dir=None):
    """
    Add static kidney boxes to an Arbuz project file
    
    Args:
        input_file: Path to input .mat file
        output_dir: Output directory (optional)
    
    Returns:
        str: Path to output file with kidney boxes
    """
    print(f"\n🔄 Adding static kidney boxes to: {os.path.basename(input_file)}")
    
    # Set up output directory
    if output_dir is None:
        # Use the correct inference directory path
        inference_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(inference_dir, f"static_kidneys_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"   📁 Output directory: {output_dir}")
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_WITH_KIDNEY_BOXES.mat")
    
    # Call MATLAB function
    matlab_cmd = [
        'matlab', '-batch',
        f"addpath('C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\src\\creatingKidneys\\src'); "
        f"add_static_kidney_boxes('{input_file}', '{output_file}'); "
        f"exit;"
    ]
    
    print(f"   🔧 Running MATLAB to add kidney boxes...")
    
    try:
        # Run MATLAB command
        result = subprocess.run(matlab_cmd, capture_output=True, text=True, 
                              cwd=os.path.dirname(__file__), encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"   ✅ MATLAB execution successful")
            # Print relevant output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if any(marker in line for marker in ['✅', '🎯', '📁', '📊', '🎨', '🟥', '🟩', 'kidney', 'box']):
                        print(f"      {line}")
        else:
            print(f"   ❌ MATLAB error: {result.stderr}")
            raise RuntimeError(f"MATLAB execution failed: {result.stderr}")
            
    except FileNotFoundError:
        raise RuntimeError("MATLAB not found. Please ensure MATLAB is installed and in PATH.")
    
    # Verify output file exists
    if os.path.exists(output_file):
        file_size_mb = os.path.getsize(output_file) / (1024*1024)
        print(f"\n✅ SUCCESS! Static kidney boxes added:")
        print(f"   📁 File: {output_file}")
        print(f"   📊 Size: {file_size_mb:.1f} MB")
        print(f"   🎯 Kidney boxes: 2 (Left and Right)")
        print(f"   👁️  Boxes will be visible as slaves in ArbuzGUI!")
        
        return output_file
    else:
        raise FileNotFoundError("Output file was not created")

def main():
    """Main execution"""
    
    # Check if file is provided as argument
    if len(sys.argv) > 1:
        input_files = [sys.argv[1]]
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]
        else:
            output_dir = None
    else:
        # Default test file
        test_dir = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training"
        input_files = [os.path.join(test_dir, "withoutROIwithMRI.mat")]
        output_dir = None
    
    try:
        results = []
        
        for input_file in input_files:
            if os.path.exists(input_file):
                print(f"\n{'='*60}")
                print(f"🔄 PROCESSING: {os.path.basename(input_file)}")
                print(f"{'='*60}")
                
                output_file = add_static_kidneys(input_file, output_dir)
                results.append(output_file)
            else:
                print(f"❌ File not found: {input_file}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📋 STATIC KIDNEY BOXES COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Processed {len(results)} files successfully")
        print(f"🎯 Static kidney boxes added to all files")
        print(f"👁️  Open files in ArbuzGUI to see kidney box slaves!")
        
        for i, result_file in enumerate(results, 1):
            print(f"{i}. {result_file}")
        
        print(f"\n🎉 Ready for ArbuzGUI - kidney boxes appear as slaves!")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
