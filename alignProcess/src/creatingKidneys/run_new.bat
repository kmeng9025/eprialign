@echo off
setlocal enabledelayedexpansion

REM AI Kidney Detection Pipeline Batch Runner
REM Usage: run.bat [--input-file <file>] [--input-dir <dir>] [--output-dir <dir>] [--help]

REM Initialize variables
set "INPUT_FILE="
set "INPUT_DIR="
set "OUTPUT_DIR="
set "SHOW_HELP="
set "DEFAULT_OUTPUT_DIR=..\..\data\inference"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--help" (
    set "SHOW_HELP=1"
    shift
    goto parse_args
)
if "%~1"=="--input-file" (
    set "INPUT_FILE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--input-dir" (
    set "INPUT_DIR=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--output-dir" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto parse_args
)
REM Skip unknown parameters
shift
goto parse_args
:end_parse

REM Show help if requested
if defined SHOW_HELP (
    echo.
    echo ========================================
    echo   AI KIDNEY DETECTION PIPELINE HELP
    echo ========================================
    echo.
    echo USAGE:
    echo   run.bat [OPTIONS]
    echo.
    echo OPTIONS:
    echo   --input-file ^<file^>    Process a single .mat file
    echo   --input-dir ^<dir^>      Process all .mat files in directory
    echo   --output-dir ^<dir^>     Output directory ^(default: %DEFAULT_OUTPUT_DIR%^)
    echo   --help                 Show this help message
    echo.
    echo EXAMPLES:
    echo   run.bat --input-file "data\test.mat"
    echo   run.bat --input-dir "data\batch" --output-dir "results"
    echo   run.bat
    echo.
    echo NOTES:
    echo   - If no parameters provided, interactive mode starts
    echo   - Cannot use both --input-file and --input-dir
    echo   - Default output directory: %DEFAULT_OUTPUT_DIR%
    echo   - Requires Python environment with required packages
    echo.
    goto end
)

REM Check for conflicting input parameters
if defined INPUT_FILE if defined INPUT_DIR (
    echo ^>^> WARNING: Both --input-file and --input-dir provided
    echo ^>^> Using --input-file: %INPUT_FILE%
    echo ^>^> Ignoring --input-dir: %INPUT_DIR%
    set "INPUT_DIR="
)

REM If no input specified, start interactive mode
if not defined INPUT_FILE if not defined INPUT_DIR (
    echo.
    echo ========================================
    echo   AI KIDNEY DETECTION PIPELINE
    echo ========================================
    echo.
    echo Please specify input:
    echo 1. Single file
    echo 2. Directory of files
    echo.
    set /p "choice=Choose option (1 or 2): "
    
    if "!choice!"=="1" (
        set /p "INPUT_FILE=Enter path to .mat file: "
    ) else if "!choice!"=="2" (
        set /p "INPUT_DIR=Enter directory path: "
    ) else (
        echo Invalid choice. Exiting.
        goto end
    )
)

REM Set default output directory if not specified
if not defined OUTPUT_DIR (
    set "OUTPUT_DIR=%DEFAULT_OUTPUT_DIR%"
)

REM Show current output directory before asking
echo.
echo Default output directory: %DEFAULT_OUTPUT_DIR%
if not defined OUTPUT_DIR (
    set /p "user_output=Enter output directory (or press Enter for default): "
    if defined user_output (
        set "OUTPUT_DIR=!user_output!"
    ) else (
        set "OUTPUT_DIR=%DEFAULT_OUTPUT_DIR%"
    )
) else (
    echo Using specified output directory: %OUTPUT_DIR%
)

REM Display configuration
echo ^>^> Mode: !if defined INPUT_FILE (Single file) else (Directory)!
if defined INPUT_FILE echo ^>^> Input file: !INPUT_FILE!
if defined INPUT_DIR echo ^>^> Input directory: !INPUT_DIR!
echo ^>^> Output directory: !OUTPUT_DIR!
echo.

REM Check if requirements are installed
echo Checking Python dependencies...
echo ========================================

REM Change to script directory
cd /d "src"

REM Check for conda environment
echo Activating conda environment...
call conda activate mrialigned
if %errorlevel% neq 0 (
    echo WARNING: Failed to activate mrialigned conda environment
    echo You may need to install the environment first
    pause
)

REM Check if key packages are installed
echo Checking required packages...
python -c "import torch, torchvision, numpy, scipy, nibabel, h5py, sklearn, skimage" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some required Python packages are missing!
    echo.
    echo A requirements.txt file is available in this directory.
    echo You can install the requirements with:
    echo   pip install -r ../requirements.txt
    echo.
    set /p "install_choice=Would you like to install missing packages now? (y/n): "
    if /i "!install_choice!"=="y" (
        echo Installing packages...
        pip install -r "../requirements.txt"
        if !errorlevel! neq 0 (
            echo FAILED to install packages. Please check your Python environment.
            pause
            goto end
        )
        echo Packages installed successfully!
    ) else (
        echo Continuing without installing packages. This may cause errors.
    )
    echo.
)

REM Run the AI kidney detection
echo.
echo Starting AI kidney detection...
echo ========================================
echo.

if defined INPUT_FILE (
    REM Single file mode
    python ai_kidney_detection.py "!INPUT_FILE!" "!OUTPUT_DIR!"
    set "PYTHON_EXIT=!errorlevel!"
) else (
    REM Directory mode - process all .mat files
    echo Processing all .mat files in directory...
    set "FILE_COUNT=0"
    set "SUCCESS_COUNT=0"
    set "ERROR_COUNT=0"
    
    for %%f in ("!INPUT_DIR!\*.mat") do (
        set /a FILE_COUNT+=1
        echo.
        echo [!FILE_COUNT!] Processing: %%~nxf
        echo ----------------------------------------
        
        python ai_kidney_detection.py "%%f" "!OUTPUT_DIR!"
        if !errorlevel! equ 0 (
            set /a SUCCESS_COUNT+=1
            echo [!FILE_COUNT!] SUCCESS: %%~nxf
        ) else (
            set /a ERROR_COUNT+=1
            echo [!FILE_COUNT!] FAILED: %%~nxf
        )
    )
    
    echo.
    echo ========================================
    echo BATCH PROCESSING SUMMARY
    echo ========================================
    echo Total files: !FILE_COUNT!
    echo Successful: !SUCCESS_COUNT!
    echo Failed: !ERROR_COUNT!
    echo.
    
    if !ERROR_COUNT! gtr 0 (
        set "PYTHON_EXIT=1"
    ) else (
        set "PYTHON_EXIT=0"
    )
)

REM Show final status
echo.
echo ========================================
if !PYTHON_EXIT! equ 0 (
    echo SUCCESS AI kidney detection completed successfully
    echo.
    echo Output files are available in: !OUTPUT_DIR!
    echo Open the processed files in ArbuzGUI to see AI-detected kidney slaves
) else (
    echo FAILED AI kidney detection encountered errors
    echo Check the output above for details
)
echo.

:end
echo Press any key to exit...
pause >nul
