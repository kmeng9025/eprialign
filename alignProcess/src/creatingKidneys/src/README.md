# AI Kidney Detection - Clean Workspace
====================================

## 🎯 Essential Files

### Core AI Pipeline
- `ai_kidney_detection.py` - Main AI kidney detection pipeline
- `unet_3d.py` - 3D U-Net model architecture
- `train_fresh_kidney.py` - Training script for new models
- `requirements.txt` - Python dependencies

### Trained Models
- `kidney_unet_model_best.pth` - Original locally trained model
- `kidney_unet_model_modal_trained.pth` - Modal AI Cloud trained model (latest)

### Modal AI Cloud Scripts
- `modal_working.py` - Primary Modal script for cloud training
- `modal_simple.py` - Simplified Modal script
- `modal_final_training.py` - Complete training pipeline
- `modal_kidney_training.py` - Alternative Modal implementation
- `modal_train_and_upload.py` - Upload and train script

### MATLAB Integration
- `create_kidney_slaves_final.m` - Creates kidney slaves in Arbuz
- `run_ai_kidney_detection_pipeline.m` - MATLAB pipeline runner
- `run_ai_kidney_pipeline_matlab.m` - Alternative MATLAB runner
- `save_ai_kidneys_matlab.m` - Saves AI results to MATLAB format
- `combine_arbuz_with_ai.m` - Combines AI results with Arbuz projects
- `create_arbuz_compatible_file.m` - Creates Arbuz-compatible files
- `create_clean_arbuz_with_kidney_masks.m` - Clean Arbuz file creation

### Utility Scripts
- `add_static_kidney_boxes.m` - Adds static kidney regions
- `compare_slaves.m` - Compare different slave implementations
- `debug_slaves.m` - Debug slave creation
- `inspect_training_slaves.m` - Inspect training data slaves
- `quick_check.m` - Quick MATLAB verification

### Documentation
- `PROJECT_SUMMARY.md` - Project overview
- `README.md` - This file

## 🚀 Usage

### Train New Model on Modal AI Cloud
```bash
cd alignProcess/src/creatingKidneys/src
modal run modal_working.py::upload_data  # Upload training data
modal run modal_working.py::run_training  # Train on A10 GPU
```

### Run AI Kidney Detection
```bash
python ai_kidney_detection.py input_file.mat [output_dir]
```

### Download Models from Modal
```bash
modal volume get kidneyCheckpoints kidney_unet_model_best.pth new_model.pth
```

## 📁 Data Structure
- Training data should be in `../../../data/training/`
- Output files are saved to `../../../data/inference/`
- Modal volumes: `kidneyDrawing` (data), `kidneyCheckpoints` (models)

## ✅ Workspace Cleaned
All debug, test, exploration, and redundant files have been removed.
Only essential production files remain.