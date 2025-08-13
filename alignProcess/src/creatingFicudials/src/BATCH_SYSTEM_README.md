# 🚀 EPRI-T AI Fiducial Detection - Batch System

## Quick Start

### Using the Batch Script (Recommended)

```batch
# Interactive mode (will prompt for missing parameters)
run.bat

# Single file processing
run.bat --input-file "C:\path\to\project.mat" --output-dir "C:\output"

# Directory processing (all .mat files)
run.bat --input-dir "C:\path\to\files" --output-dir "C:\output"

# Show help
run.bat --help
```

### Direct Python Usage

```bash
# Single file
python simple_box_fiducials.py --input_file "project.mat" --output_dir "output"

# Directory processing
python simple_box_fiducials.py --input_dir "../data/training" --output_dir "results"

# Interactive mode
python simple_box_fiducials.py
```

## 📁 File Structure

```
src/
├── run.bat                      # 🎯 Main batch launcher
├── simple_box_fiducials.py      # 🤖 AI detection system
├── fiducial_model.py            # 🧠 AI model definition
├── predict_fiducials.py         # 🔍 Standalone inference
├── train_fiducial_model.py      # 📚 Model training
└── prep_fiducial_data.py        # 🔧 Data preprocessing
```

## 🎮 Usage Examples

### Example 1: Process Single Project File

```batch
run.bat --input-file "C:\EPRI-T\data\withoutROI.mat" --output-dir "C:\results"
```

**What happens:**
1. ✅ Loads AI model (if available)
2. 🔍 Detects fiducials in all images using AI
3. 📦 Falls back to static boxes if AI fails
4. 💾 Creates enhanced .mat file with fiducials
5. 🔧 Generates MATLAB integration script
6. ▶️ Runs ArbuzGUI integration

### Example 2: Batch Process Directory

```batch
run.bat --input-dir "C:\EPRI-T\data\training" --output-dir "C:\batch_results"
```

**What happens:**
1. 📂 Finds all .mat files in directory
2. 🔄 Processes each file individually
3. 📊 Provides summary statistics
4. 📁 Creates timestamped output folders

### Example 3: Interactive Mode with Defaults

```batch
run.bat
```

**Interactive prompts with smart defaults:**
```
No input specified. Please provide either:
  1. Single file path
  2. Directory path

Enter choice (1 or 2): 1
Enter full path to .mat file: C:\data\project.mat

Output directory not specified.
Default: ../data/inference

Enter output directory (or press Enter for default): [Enter]
Using default output directory: ../data/inference
```

**Benefits:**
- ⚡ **Quick Testing**: Just press Enter to use the standard inference folder
- 🎯 **Consistent Structure**: Follows your project's folder organization
- 🔧 **Still Flexible**: Can specify custom paths when needed

## ⚙️ Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input-file` | Single .mat file to process | `"C:\data\project.mat"` |
| `--input-dir` | Directory with .mat files | `"C:\data\training"` |
| `--output-dir` | Output directory | `"C:\results"` |
| `--help` | Show help message | N/A |

## 🤖 AI Detection Features

- **Automatic AI Loading**: Uses trained model from `../models/fiducial_model.pth`
- **Smart Fallback**: Falls back to static boxes if AI fails
- **Threshold Control**: Default 0.5, optimal for current model
- **Device Selection**: Automatically uses CUDA if available
- **Single Slave Mode**: Creates one "AI Fiducials" mask per image

## 📊 Output Structure

Each run creates its own unique timestamped folder with automatic cleanup:

```
output_directory/
├── fiducials_project_20250806_143022/         # Single file run
│   └── project_with_fiducials.mat             # ✅ Final enhanced project file
├── batch_training_20250806_144511/            # Directory batch run
│   ├── file1_with_fiducials.mat               # ✅ Final enhanced project file
│   ├── file2_with_fiducials.mat               # ✅ Final enhanced project file
│   └── file3_with_fiducials.mat               # ✅ Final enhanced project file
└── fiducials_another_20250806_150033/         # Another single file run
    └── another_with_fiducials.mat              # ✅ Final enhanced project file
```

**Automatic Cleanup Process:**
1. 📝 Creates intermediate files during processing (`box_fiducial_data.mat`, `add_fiducials_manual.m`)
2. 🔧 Runs MATLAB integration to create final enhanced project file
3. 🧹 **Automatically removes intermediate files**, keeping only the final `.mat` file
4. 💎 **Result**: Clean output with only the enhanced project file you need

**Key Benefits:**
- ✅ **No Overwriting**: Each run gets its own unique folder
- 🕒 **Timestamped**: Easy to identify when runs occurred  
- 🧹 **Auto-Cleanup**: Intermediate files removed automatically
- 💎 **Clean Results**: Only final enhanced project files remain
- 🔍 **Traceable**: Full history of all processing runs

## 🎯 Current AI Performance

- ✅ **30 fiducials detected** across 7 EPR images
- 🎪 **4-6 fiducials per image** (variable based on content)
- 📏 **Volume range**: 1,048-1,842 voxels per detection
- 🚀 **Processing speed**: ~2-5 seconds per 64³ volume

## 🔧 Advanced Options

### Python Script Advanced Parameters

```bash
python simple_box_fiducials.py \
  --input_file "project.mat" \
  --output_dir "results" \
  --threshold 0.4 \              # AI detection sensitivity
  --device cuda \                # Force GPU usage
  --num_boxes 2 \                # Fallback box count
  --box_size 3                   # Fallback box size
```

### Model Path Customization

```bash
python simple_box_fiducials.py \
  --input_file "project.mat" \
  --model_path "custom_model.pth" \
  --output_dir "results"
```

## ✅ System Requirements

- **Windows**: PowerShell 5.1+ (built-in on Windows 10/11)
- **Python**: 3.7+ with PyTorch, scipy, numpy
- **MATLAB**: For ArbuzGUI integration (optional)
- **RAM**: 8GB+ recommended for AI processing
- **Storage**: ~2GB for models and data

## 🚨 Error Handling

The system gracefully handles:
- ❌ **Missing files**: Clear error messages
- 🔄 **AI model failures**: Automatic fallback to static boxes
- 📁 **Directory creation**: Auto-creates output directories
- 🛡️ **File validation**: Checks .mat file existence
- 📊 **Batch processing**: Continues on individual file errors

## 🎉 Success Example

```
🚀 Starting ArbuzGUI fiducial integration
   Input directory: C:\EPRI-T\data\training
   Found 7 .mat files
   Output: C:\results
   Device: cuda
   🤖 Loading AI model: ../models/fiducial_model.pth
   ✅ Loaded AI model from ../models/fiducial_model.pth
   Best validation loss: 0.5375
   🎯 AI threshold: 0.5

📂 Processing: withoutROI.mat
🤖 AI detected 30 fiducials across 7 images
✅ Successfully processed withoutROI.mat

🎯 OVERALL SUMMARY:
   Files processed: 1/1
   Total images: 7 (0 MRI, 7 EPR)
   Total fiducials: 30
   AI detections: 7 images
   Box fallbacks: 0 images
   Output directory: C:\results

✅ Batch fiducial integration completed using AI + fallback boxes!
```

## 🎪 Ready to Use!

Your AI fiducial detection system is now fully integrated with a user-friendly batch interface! 🎯
