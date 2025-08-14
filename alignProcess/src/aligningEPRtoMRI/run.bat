@echo off
REM EPR-to-MRI Alignment Pipeline
REM This script runs the complete alignment pipeline using fiducials
REM Usage: run.bat [input_file.mat]
REM If no input file is provided, uses the default training file

setlocal enabledelayedexpansion

echo =====================================
echo EPR-to-MRI Alignment Pipeline
echo =====================================
echo.

REM Check if input file is provided
set INPUT_FILE=%1
if "%INPUT_FILE%"=="" (
    echo.
    echo Please enter the path to your input .mat file:
    echo (or press Enter to use default training data)
    set /p USER_INPUT="Input file path: "
    
    if "!USER_INPUT!"=="" (
        set INPUT_FILE=C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat
        echo Using default: !INPUT_FILE!
    ) else (
        set INPUT_FILE=!USER_INPUT!
        echo Using: !INPUT_FILE!
    )
) else (
    echo Input file: %INPUT_FILE%
)

echo.

REM Check if input file exists
if not exist "%INPUT_FILE%" (
    echo ERROR: Input file does not exist: %INPUT_FILE%
    echo.
    echo Usage: run.bat [input_file.mat]
    echo Example: run.bat "C:\path\to\your\data.mat"
    pause
    exit /b 1
)

REM Get current timestamp for output naming
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "timestamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"

REM Create output directory
set "OUTPUT_DIR=%~dp0output\alignment_%timestamp%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Output directory: %OUTPUT_DIR%
echo.

REM Step 1: Check MATLAB availability
echo Step 1: Checking MATLAB availability...
matlab -batch "fprintf('MATLAB is available\n'); exit" >nul 2>&1
if errorlevel 1 (
    echo ERROR: MATLAB is not available or not in PATH
    echo Please ensure MATLAB is installed and accessible from command line
    pause
    exit /b 1
)
echo MATLAB is available.
echo.

REM Step 2: Check if fiducials detection is needed
echo Step 2: Checking for existing fiducials...

REM Look for existing fiducials file
set "FIDUCIALS_FILE="
for /f "delims=" %%i in ('dir /s /b "%~dp0..\..\data\inference\*with_boxes.mat" 2^>nul') do (
    set "FIDUCIALS_FILE=%%i"
    goto found_fiducials
)

:found_fiducials
if "%FIDUCIALS_FILE%"=="" (
    echo No existing fiducials found. Running fiducial detection first...
    
    REM Run fiducials detection
    set "FIDUCIALS_SCRIPT=%~dp0..\creatingFiducials\run.bat"
    if not exist "!FIDUCIALS_SCRIPT!" (
        echo ERROR: Fiducials detection script not found: !FIDUCIALS_SCRIPT!
        pause
        exit /b 1
    )
    
    echo Running: "!FIDUCIALS_SCRIPT!" "%INPUT_FILE%"
    call "!FIDUCIALS_SCRIPT!" "%INPUT_FILE%"
    if errorlevel 1 (
        echo ERROR: Fiducials detection failed
        pause
        exit /b 1
    )
    
    REM Find the newly created fiducials file
    for /f "delims=" %%i in ('dir /s /b "%~dp0..\..\data\inference\*with_boxes.mat" 2^>nul') do (
        set "FIDUCIALS_FILE=%%i"
    )
) else (
    echo Found existing fiducials: %FIDUCIALS_FILE%
)

if "%FIDUCIALS_FILE%"=="" (
    echo ERROR: Could not find or create fiducials file
    pause
    exit /b 1
)

echo Using fiducials file: %FIDUCIALS_FILE%
echo.

REM Step 3: Run alignment
echo Step 3: Running EPR-to-MRI alignment...

REM Create temporary MATLAB script
set "TEMP_SCRIPT=%OUTPUT_DIR%\run_alignment.m"
echo Creating alignment script: %TEMP_SCRIPT%

(
echo %% Temporary alignment script with all functions included
echo addpath^('%~dp0'^);
echo.
echo fprintf^('Starting EPR-to-MRI alignment...\n'^);
echo.
echo %% File paths
echo input_file = '%INPUT_FILE:\=\\%';
echo fiducials_file = '%FIDUCIALS_FILE:\=\\%';
echo output_file = '%OUTPUT_DIR:\=\\%\aligned_data.mat';
echo.
echo fprintf^('Input file: %%s\n', input_file^);
echo fprintf^('Fiducials file: %%s\n', fiducials_file^);
echo fprintf^('Output file: %%s\n', output_file^);
echo.
echo %% Load fiducials data
echo fprintf^('Loading fiducials data...\n'^);
echo fid_data = load^(fiducials_file^);
echo.
echo %% Extract fiducials
echo fiducials = extract_fiducials_from_data_inline^(fid_data^);
echo.
echo %% Display fiducials summary  
echo display_fiducials_summary_inline^(fiducials^);
echo.
echo %% Calculate alignment transformations
echo fprintf^('\nCalculating alignment transformations...\n'^);
echo reference_image = 'MRI';
echo epr_images = {'PreHemo', 'Hemo', 'Post_Tranfus', 'Post_Transfus2'};
echo.
echo transformations = {};
echo for i = 1:length^(epr_images^)
echo     epr_name = epr_images{i};
echo     if isfield^(fiducials, epr_name^) ^&^& isfield^(fiducials, 'MRI'^)
echo         fprintf^('  Aligning %%s to %%s...\n', epr_name, 'MRI'^);
echo         source_points = fiducials.^(epr_name^);
echo         target_points = fiducials.^('MRI'^);
echo         if size^(source_points, 1^) ^>= 3 ^&^& size^(target_points, 1^) ^>= 3
echo             transform = calculate_fiducial_alignment_inline^(source_points, target_points^);
echo             transformations{end+1} = struct^('name', ['^>' epr_name], 'matrix', transform, 'target', '^>MRI'^);
echo             fprintf^('    Successfully calculated transformation for %%s\n', epr_name^);
echo         else
echo             fprintf^('    Warning: Not enough fiducials for %%s ^(need at least 3^)\n', epr_name^);
echo         end
echo     else
echo         fprintf^('    Warning: No fiducials found for %%s\n', epr_name^);
echo     end
echo end
echo.
echo %% Apply transformations and save result
echo fprintf^('\nApplying transformations...\n'^);
echo apply_transformations_to_data_inline^(fid_data, transformations, output_file^);
echo.
echo fprintf^('Alignment complete! Output saved to: %%s\n', output_file^);
echo.
echo %% Generate summary report
echo report_file = '%OUTPUT_DIR:\=\\%\alignment_report.txt';
echo generate_alignment_report_inline^(transformations, report_file^);
echo.
echo fprintf^('Alignment pipeline completed successfully!\n'^);
echo.
echo %%%% INLINE FUNCTIONS %%%%
echo.
echo function fiducials = extract_fiducials_from_data_inline^(data^)
echo     fiducials = struct^(^);
echo     for i = 1:length^(data.images^)
echo         img = data.images{i};
echo         img_name = clean_image_name_inline^(img.Name^);
echo         if isfield^(img, 'slaves'^) ^&^& ~isempty^(img.slaves^)
echo             slave = img.slaves{1};
echo             if isfield^(slave, 'data'^) ^&^& ~isempty^(slave.data^)
echo                 mask = slave.data;
echo                 cc = bwconncomp^(mask^);
echo                 if cc.NumObjects ^> 0
echo                     points = zeros^(cc.NumObjects, 3^);
echo                     for j = 1:cc.NumObjects
echo                         [y, x, z] = ind2sub^(size^(mask^), cc.PixelIdxList{j}^);
echo                         points^(j, :^) = [mean^(x^), mean^(y^), mean^(z^)];
echo                     end
echo                     fiducials.^(img_name^) = points;
echo                     fprintf^('  Found %%d fiducials for %%s\n', size^(points, 1^), img_name^);
echo                 else
echo                     fprintf^('  No fiducials detected for %%s\n', img_name^);
echo                 end
echo             end
echo         end
echo     end
echo end
echo.
echo function clean_name = clean_image_name_inline^(name^)
echo     clean_name = name;
echo     if startsWith^(clean_name, '^>'^)
echo         clean_name = clean_name^(2:end^);
echo     end
echo     clean_name = matlab.lang.makeValidName^(clean_name^);
echo end
echo.
echo function display_fiducials_summary_inline^(fiducials^)
echo     fprintf^('\nFIDUCIALS SUMMARY:\n'^);
echo     field_names = fieldnames^(fiducials^);
echo     for i = 1:length^(field_names^)
echo         name = field_names{i};
echo         points = fiducials.^(name^);
echo         fprintf^('  %%s: %%d fiducials\n', name, size^(points, 1^)^);
echo         if size^(points, 1^) ^> 0
echo             fprintf^('    Example points:\n'^);
echo             for j = 1:min^(3, size^(points, 1^)^)
echo                 fprintf^('      [%%.2f, %%.2f, %%.2f]\n', points^(j, 1^), points^(j, 2^), points^(j, 3^)^);
echo             end
echo         end
echo     end
echo end
echo.
echo function transform_matrix = calculate_fiducial_alignment_inline^(source_points, target_points^)
echo     n_pairs = min^(size^(source_points, 1^), size^(target_points, 1^)^);
echo     src = source_points^(1:n_pairs, :^);
echo     tgt = target_points^(1:n_pairs, :^);
echo     fprintf^('      Using %%d point pairs for alignment\n', n_pairs^);
echo     src_centroid = mean^(src, 1^);
echo     tgt_centroid = mean^(tgt, 1^);
echo     src_centered = src - src_centroid;
echo     tgt_centered = tgt - tgt_centroid;
echo     src_scale = sqrt^(sum^(src_centered^(:^).^^2^) / n_pairs^);
echo     tgt_scale = sqrt^(sum^(tgt_centered^(:^).^^2^) / n_pairs^);
echo     scale_factor = tgt_scale / src_scale;
echo     fprintf^('      Scale factor: %%.4f\n', scale_factor^);
echo     src_scaled = src_centered * scale_factor;
echo     H = src_scaled' * tgt_centered;
echo     [U, ~, V] = svd^(H^);
echo     R = V * U';
echo     if det^(R^) ^< 0
echo         V^(:, 3^) = -V^(:, 3^);
echo         R = V * U';
echo     end
echo     t = tgt_centroid' - R * ^(src_centroid * scale_factor^)';
echo     transform_matrix = eye^(4^);
echo     transform_matrix^(1:3, 1:3^) = R * scale_factor;
echo     transform_matrix^(1:3, 4^) = t;
echo     src_transformed = ^(src * scale_factor * R' + t'^)';
echo     error_distances = sqrt^(sum^(^(src_transformed' - tgt^).^^2, 2^)^);
echo     mean_error = mean^(error_distances^);
echo     max_error = max^(error_distances^);
echo     fprintf^('      Mean alignment error: %%.4f\n', mean_error^);
echo     fprintf^('      Max alignment error: %%.4f\n', max_error^);
echo end
echo.
echo function apply_transformations_to_data_inline^(original_data, transformations, output_file^)
echo     new_data = original_data;
echo     for i = 1:length^(transformations^)
echo         trans = transformations{i};
echo         img_name = trans.name;
echo         transform_matrix = trans.matrix;
echo         for j = 1:length^(new_data.images^)
echo             img = new_data.images{j};
echo             if strcmp^(img.Name, img_name^)
echo                 new_data.images{j}.A = transform_matrix * img.A;
echo                 fprintf^('  Applying transformation to %%s\n', img_name^);
echo                 break;
echo             end
echo         end
echo     end
echo     fprintf^('  Saving aligned data to: %%s\n', output_file^);
echo     save^(output_file, '-struct', 'new_data'^);
echo end
echo.
echo function generate_alignment_report_inline^(transformations, report_file^)
echo     fprintf^('Generating alignment report: %%s\n', report_file^);
echo     fid = fopen^(report_file, 'w'^);
echo     if fid == -1
echo         error^('Could not create report file: %%s', report_file^);
echo     end
echo     fprintf^(fid, 'EPR-to-MRI Alignment Report\n'^);
echo     fprintf^(fid, '===========================\n\n'^);
echo     fprintf^(fid, 'Generated on: %%s\n\n', datestr^(now^)^);
echo     fprintf^(fid, 'Summary:\n'^);
echo     fprintf^(fid, '--------\n'^);
echo     fprintf^(fid, 'Number of aligned images: %%d\n\n', length^(transformations^)^);
echo     if ~isempty^(transformations^)
echo         fprintf^(fid, 'Transformation Details:\n'^);
echo         fprintf^(fid, '----------------------\n\n'^);
echo         for i = 1:length^(transformations^)
echo             trans = transformations{i};
echo             T = trans.matrix;
echo             fprintf^(fid, '%%s:\n', trans.name^);
echo             scale_x = norm^(T^(1:3, 1^)^);
echo             scale_y = norm^(T^(1:3, 2^)^);
echo             scale_z = norm^(T^(1:3, 3^)^);
echo             translation = T^(1:3, 4^);
echo             fprintf^(fid, '  Scale factors: [%%.6f, %%.6f, %%.6f]\n', scale_x, scale_y, scale_z^);
echo             fprintf^(fid, '  Translation: [%%.4f, %%.4f, %%.4f]\n', translation^(1^), translation^(2^), translation^(3^)^);
echo             fprintf^(fid, '\n'^);
echo         end
echo     end
echo     fclose^(fid^);
echo     fprintf^('Report saved successfully.\n'^);
echo end
) > "%TEMP_SCRIPT%"

REM Run MATLAB script
echo Running MATLAB alignment script...
cd /d "%~dp0"
matlab -batch "run('%TEMP_SCRIPT:\=\\%')"

if errorlevel 1 (
    echo ERROR: Alignment failed
    pause
    exit /b 1
)

REM Clean up temporary script
del "%TEMP_SCRIPT%" >nul 2>&1

echo.
echo Step 4: Generating summary...

REM Create summary file (only keeping the aligned data and report)
set "SUMMARY_FILE=%OUTPUT_DIR%\summary.txt"
(
echo EPR-to-MRI Alignment Summary
echo =============================
echo.
echo Timestamp: %date% %time%
echo Input file: %INPUT_FILE%
echo Fiducials file: %FIDUCIALS_FILE%
echo Output directory: %OUTPUT_DIR%
echo.
echo Main output file:
echo - aligned_data.mat ^(load this in ArbuzGUI^)
echo.
echo Additional files:
echo - alignment_report.txt ^(detailed statistics^)
echo - summary.txt ^(this file^)
echo.
echo Status: SUCCESS
) > "%SUMMARY_FILE%"

echo.
echo =====================================
echo EPR-to-MRI Alignment Complete!
echo =====================================
echo.
echo Output directory: %OUTPUT_DIR%
echo.
echo Main result: aligned_data.mat
echo ^(Load this file in ArbuzGUI to view the aligned EPR images^)
echo.
echo Additional files:
echo - alignment_report.txt ^(detailed alignment statistics^)
echo - summary.txt ^(processing summary^)
echo.

pause
