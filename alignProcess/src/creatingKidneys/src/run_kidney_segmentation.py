#!/usr/bin/env python3
"""
Command-line interface for the kidney segmentation pipeline

Usage examples:
    python run_kidney_segmentation.py input.mat
    python run_kidney_segmentation.py input.mat --output-dir ./results
    python run_kidney_segmentation.py --help
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from kidney_segmentation_pipeline import KidneySegmentationPipeline

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description="AI-powered kidney segmentation for MRI images in Arbuz project files",
        epilog="""
Examples:
  python run_kidney_segmentation.py data/input.mat
  python run_kidney_segmentation.py data/input.mat --output-dir results/
  python run_kidney_segmentation.py ../data/training/withoutROIwithMRI.mat
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input .mat project file containing MRI and EPR data"
    )
    
    parser.add_argument(
        "--output-dir",
        default="../../../data/inference",
        help="Output directory for results (default: ../../../data/inference)"
    )
    
    parser.add_argument(
        "--model-path",
        help="Path to custom nnU-Net model (optional)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_file = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Validate input file
    if not input_file.exists():
        print(f"Error: Input file does not exist: {input_file}")
        return 1
    
    if not input_file.suffix.lower() == '.mat':
        print(f"Error: Input file must be a .mat file: {input_file}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("🔬 Kidney Segmentation Pipeline")
        print("=" * 50)
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}")
        if args.model_path:
            print(f"Model path: {args.model_path}")
        print()
        
        # Create and run pipeline
        pipeline = KidneySegmentationPipeline(
            input_file=str(input_file),
            output_dir=str(output_dir),
            model_path=args.model_path
        )
        
        pipeline.run()
        
        print()
        print("✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {pipeline.run_output_dir}")
        
        # List output files
        output_files = list(pipeline.run_output_dir.glob("*.mat"))
        if output_files:
            print("📋 Output files:")
            for file in output_files:
                print(f"   {file.name}")
        
        print()
        print("🔍 Next steps:")
        print("   1. Open MATLAB and navigate to Arbuz2.0 directory")
        print("   2. Run 'ArbuzGUI' to launch the viewer")
        print("   3. Open the output .mat file to view kidney masks")
        print("   4. Kidney masks will appear as slave images under MRI images")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
