% Temporary alignment script
addpath('C:\Users\ftmen\Documents\mrialign\alignProcess\src\aligningEPRtoMRI\');

fprintf('Starting EPR-to-MRI alignment...\n');

% File paths
input_file = 'C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\data\\training\\withoutROIwithMRI.mat';
fiducials_file = 'C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\data\\inference\\fiducials_withoutROIwithMRIwithoutTransformations_20250814_125342\\box_fiducials_withoutROIwithMRIwithoutTransformations_20250814_125345\\withoutROIwithMRIwithoutTransformations_with_boxes.mat';
output_file = 'C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\src\\aligningEPRtoMRI\\output\\alignment_20250814_140722\aligned_data.mat';

fprintf('Input file: %s\n', input_file);
fprintf('Fiducials file: %s\n', fiducials_file);
fprintf('Output file: %s\n', output_file);

% Load fiducials data
fprintf('Loading fiducials data...\n');
fid_data = load(fiducials_file);

% Extract fiducials
fiducials = extract_fiducials_from_data(fid_data);

% Display fiducials summary
display_fiducials_summary(fiducials);

% Calculate alignment transformations
fprintf('\nCalculating alignment transformations...\n');
reference_image = 'MRI';
epr_images = {'PreHemo', 'Hemo', 'Post_Tranfus', 'Post_Transfus2'};

transformations = {};
for i = 1:length(epr_images)
    epr_name = epr_images{i};
    if isfield(fiducials, epr_name) && isfield(fiducials, 'MRI')
        fprintf('  Aligning %s to %s...\n', epr_name, 'MRI');
        source_points = fiducials.(epr_name);
        target_points = fiducials.MRI;
        if size(source_points, 1) >= 3 && size(target_points, 1) >= 3
            transform = calculate_fiducial_alignment(source_points, target_points);
            fprintf('    Successfully calculated transformation for %s\n', epr_name);
        else
            fprintf('    Warning: Not enough fiducials for %s (need at least 3)\n', epr_name);
        end
    else
        fprintf('    Warning: No fiducials found for %s\n', epr_name);
    end
end

% Apply transformations and save result
fprintf('\nApplying transformations...\n');
apply_transformations_to_data(fid_data, transformations, output_file);

fprintf('Alignment complete %s\n', output_file);

% Generate summary report
report_file = 'C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\src\\aligningEPRtoMRI\\output\\alignment_20250814_140722\alignment_report.txt';
generate_alignment_report(transformations, report_file);

fprintf('Alignment pipeline completed successfully\n');
