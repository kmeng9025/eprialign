@echo off
setlocal enabledelayedexpansion

REM ========================================
REM  EPRI-T AI Fiducial Detection System
REM  Batch Script Launcher
REM ========================================

echo.
echo ========================================
echo  EPRI-T AI Fiducial Detection System
echo ========================================
echo.

REM Initialize variables
set INPUT_FILE=
set INPUT_DIR=
set OUTPUT_DIR=
set SHOW_HELP=0

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_parsed
if /i "%~1"=="--input-file" (
    set INPUT_FILE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--input-dir" (
    set INPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--output-dir" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    set SHOW_HELP=1
    goto :show_help
)
if /i "%~1"=="-h" (
    set SHOW_HELP=1
    goto :show_help
)
echo Error: Unknown parameter "%~1"
goto :show_help

:args_parsed

REM Show help if requested
if %SHOW_HELP%==1 goto :show_help

REM Validate mutually exclusive input options
if defined INPUT_FILE if defined INPUT_DIR (
    echo Error: Cannot specify both --input-file and --input-dir
    echo Use --input-file for single file processing OR --input-dir for batch processing
    echo.
    goto :show_help
)

REM Prompt for input if neither provided
if not defined INPUT_FILE if not defined INPUT_DIR (
    echo No input specified. Please provide either:
    echo   1. Single file path  2. Directory path
    echo.
    set /p "user_choice=Enter choice (1 or 2): "
    
    if "!user_choice!"=="1" (
        set /p "INPUT_FILE=Enter full path to .mat file: "
        if not exist "!INPUT_FILE!" (
            echo Error: File "!INPUT_FILE!" does not exist
            @REM pause
            exit /b 1
        )
    ) else if "!user_choice!"=="2" (
        set /p "INPUT_DIR=Enter directory path containing .mat files: "
        if not exist "!INPUT_DIR!" (
            echo Error: Directory "!INPUT_DIR!" does not exist
            @REM pause
            exit /b 1
        )
    ) else (
        echo Error: Invalid choice. Please enter 1 or 2
        @REM pause
        exit /b 1
    )
)

REM Prompt for output directory if not provided
if not defined OUTPUT_DIR (
    echo.
    echo Output directory not specified.
    echo Default: ..\..\data\inference
    echo.
    set /p "OUTPUT_DIR=Enter output directory (or press Enter for default): "
    
    REM If user just pressed Enter, use default
    if "!OUTPUT_DIR!"=="" (
        set OUTPUT_DIR=..\..\data\inference
        echo Using default output directory: !OUTPUT_DIR!
    )
)

REM Create output directory if it doesn't exist
if not exist "!OUTPUT_DIR!" (
    echo Creating output directory: !OUTPUT_DIR!
    mkdir "!OUTPUT_DIR!" 2>nul
    if errorlevel 1 (
        echo Error: Could not create output directory "!OUTPUT_DIR!"
        @REM pause
        exit /b 1
    )
)

REM Display configuration
echo.
echo Configuration:
if defined INPUT_FILE (
    echo   Input file: !INPUT_FILE!
) else (
    echo   Input directory: !INPUT_DIR!
)
echo   Output directory: !OUTPUT_DIR!
echo.

REM Build Python command
set PYTHON_CMD=python ./src/simple_box_fiducials.py

if defined INPUT_FILE (
    set PYTHON_CMD=!PYTHON_CMD! --input_file "!INPUT_FILE!"
) else (
    set PYTHON_CMD=!PYTHON_CMD! --input_dir "!INPUT_DIR!"
)

set PYTHON_CMD=!PYTHON_CMD! --output_dir "!OUTPUT_DIR!"

REM Execute the Python script
echo Starting AI fiducial detection...
echo Command: !PYTHON_CMD!
echo.
!PYTHON_CMD!

REM Check exit code
if errorlevel 1 (
    echo.
    echo ❌ Error: Fiducial detection failed
    @REM pause
    exit /b 1
) else (
    echo.
    echo ✅ Success: Fiducial detection completed
    echo 📂 Results saved in timestamped folder within: !OUTPUT_DIR!
    echo 💡 Each run creates its own unique timestamped subfolder to prevent overwriting
)

echo.
@REM pause
exit /b 0

:show_help
echo Usage: run.bat [OPTIONS]
echo.
echo AI Fiducial Detection System - Batch Launcher
echo.
echo OPTIONS:
echo   --input-file PATH    Process a single .mat file
echo   --input-dir PATH     Process all .mat files in directory
echo   --output-dir PATH    Output directory for results
echo   --help, -h          Show this help message
echo.
echo EXAMPLES:
echo   run.bat --input-file "C:\data\sample.mat" --output-dir "C:\output"
echo   run.bat --input-dir "C:\data" --output-dir "C:\output"
echo   run.bat                                    (interactive mode)
echo.
echo NOTES:
echo   • --input-file and --input-dir are mutually exclusive
echo   • If no input is specified, you'll be prompted interactively
echo   • If no output directory is specified, defaults to ../../data/inference
echo   • Output directory will be created if it doesn't exist
echo   • Each run creates a unique timestamped subfolder
echo.
@REM pause
exit /b 0
