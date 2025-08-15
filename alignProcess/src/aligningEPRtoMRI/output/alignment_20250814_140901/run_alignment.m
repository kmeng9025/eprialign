% Temporary alignment script with all functions included
addpath('C:\Users\ftmen\Documents\mrialign\alignProcess\src\aligningEPRtoMRI\');

fprintf('Starting EPR-to-MRI alignment...\n');

% File paths
input_file = 'C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\data\\training\\withoutROIwithMRI.mat';
fiducials_file = 'C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\data\\inference\\fiducials_withoutROIwithMRIwithoutTransformations_20250814_125342\\box_fiducials_withoutROIwithMRIwithoutTransformations_20250814_125345\\withoutROIwithMRIwithoutTransformations_with_boxes.mat';
output_file = 'C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\src\\aligningEPRtoMRI\\output\\alignment_20250814_140901\aligned_data.mat';

fprintf('Input file: %s\n', input_file);
fprintf('Fiducials file: %s\n', fiducials_file);
fprintf('Output file: %s\n', output_file);

% Load fiducials data
fprintf('Loading fiducials data...\n');
fid_data = load(fiducials_file);

% Extract fiducials
fiducials = extract_fiducials_from_data_inline(fid_data);

% Display fiducials summary  
display_fiducials_summary_inline(fiducials);

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
            transform = calculate_fiducial_alignment_inline(source_points, target_points);
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
apply_transformations_to_data_inline(fid_data, transformations, output_file);

fprintf('Alignment complete %s\n', output_file);

% Generate summary report
report_file = 'C:\\Users\\ftmen\\Documents\\mrialign\\alignProcess\\src\\aligningEPRtoMRI\\output\\alignment_20250814_140901\alignment_report.txt';
generate_alignment_report_inline(transformations, report_file);

fprintf('Alignment pipeline completed successfully\n');

%% INLINE FUNCTIONS %%

function fiducials = extract_fiducials_from_data_inline(data)
    fiducials = struct();
    for i = 1:length(data.images)
        img = data.images{i};
        img_name = clean_image_name_inline(img.Name);
        if isfield(img, 'slaves') && ~isempty(img.slaves)
            slave = img.slaves{1};
            if isfield(slave, 'data') && ~isempty(slave.data)
                mask = slave.data;
                cc = bwconncomp(mask);
                if cc.NumObjects > 0
                    points = zeros(cc.NumObjects, 3);
                    for j = 1:cc.NumObjects
                        [y, x, z] = ind2sub(size(mask), cc.PixelIdxList{j});
                        points(j, :) = [mean(x), mean(y), mean(z)];
                    end
                    fiducials.(img_name) = points;
                    fprintf('  Found %d fiducials for %s\n', size(points, 1), img_name);
                else
                    fprintf('  No fiducials detected for %s\n', img_name);
                end
            end
        end
    end
end

function clean_name = clean_image_name_inline(name)
    clean_name = name;
    if startsWith(clean_name, '>')
        clean_name = clean_name(2:end);
    end
    clean_name = matlab.lang.makeValidName(clean_name);
end

function display_fiducials_summary_inline(fiducials)
    fprintf('\nFIDUCIALS SUMMARY:\n');
    field_names = fieldnames(fiducials);
    for i = 1:length(field_names)
        name = field_names{i};
        points = fiducials.(name);
        fprintf('  %s: %d fiducials\n', name, size(points, 1));
        if size(points, 1) > 0
            fprintf('    Example points:\n');
            for j = 1:min(3, size(points, 1))
                fprintf('      [%.2f, %.2f, %.2f]\n', points(j, 1), points(j, 2), points(j, 3));
            end
        end
    end
end

function transform_matrix = calculate_fiducial_alignment_inline(source_points, target_points)
    n_pairs = min(size(source_points, 1), size(target_points, 1));
    src = source_points(1:n_pairs, :);
    tgt = target_points(1:n_pairs, :);
    fprintf('      Using %d point pairs for alignment\n', n_pairs);
    src_centroid = mean(src, 1);
    tgt_centroid = mean(tgt, 1);
    src_centered = src - src_centroid;
    tgt_centered = tgt - tgt_centroid;
    src_scale = sqrt(sum(src_centered(:).^2) / n_pairs);
    tgt_scale = sqrt(sum(tgt_centered(:).^2) / n_pairs);
    scale_factor = tgt_scale / src_scale;
    fprintf('      Scale factor: %.4f\n', scale_factor);
    src_scaled = src_centered * scale_factor;
    H = src_scaled' * tgt_centered;
    [U, ~, V] = svd(H);
    R = V * U';
    if det(R) < 0
        V(:, 3) = -V(:, 3);
        R = V * U';
    end
    t = tgt_centroid' - R * (src_centroid * scale_factor)';
    transform_matrix = eye(4);
    transform_matrix(1:3, 1:3) = R * scale_factor;
    transform_matrix(1:3, 4) = t;
    src_transformed = (src * scale_factor * R' + t')';
    error_distances = sqrt(sum((src_transformed' - tgt).^2, 2));
    mean_error = mean(error_distances);
    max_error = max(error_distances);
    fprintf('      Mean alignment error: %.4f\n', mean_error);
    fprintf('      Max alignment error: %.4f\n', max_error);
end

function apply_transformations_to_data_inline(original_data, transformations, output_file)
    new_data = original_data;
    for i = 1:length(transformations)
        trans = transformations{i};
        img_name = trans.name;
        transform_matrix = trans.matrix;
        for j = 1:length(new_data.images)
            img = new_data.images{j};
            if strcmp(img.Name, img_name)
                new_data.images{j}.A = transform_matrix * img.A;
                fprintf('  Applying transformation to %s\n', img_name);
                break;
            end
        end
    end
    fprintf('  Saving aligned data to: %s\n', output_file);
    save(output_file, '-struct', 'new_data');
end

function generate_alignment_report_inline(transformations, report_file)
    fprintf('Generating alignment report: %s\n', report_file);
    fid = fopen(report_file, 'w');
    if fid == -1
        error('Could not create report file: %s', report_file);
    end
    fprintf(fid, 'EPR-to-MRI Alignment Report\n');
    fprintf(fid, '===========================\n\n');
    fprintf(fid, 'Generated on: %s\n\n', datestr(now));
    fprintf(fid, 'Summary:\n');
    fprintf(fid, '--------\n');
    fprintf(fid, 'Number of aligned images: %d\n\n', length(transformations));
    if ~isempty(transformations)
        fprintf(fid, 'Transformation Details:\n');
        fprintf(fid, '----------------------\n\n');
        for i = 1:length(transformations)
            trans = transformations{i};
            T = trans.matrix;
            fprintf(fid, '%s:\n', trans.name);
            scale_x = norm(T(1:3, 1));
            scale_y = norm(T(1:3, 2));
            scale_z = norm(T(1:3, 3));
            translation = T(1:3, 4);
            fprintf(fid, '  Scale factors: [%.6f, %.6f, %.6f]\n', scale_x, scale_y, scale_z);
            fprintf(fid, '  Translation: [%.4f, %.4f, %.4f]\n', translation(1), translation(2), translation(3));
            fprintf(fid, '\n');
        end
    end
    fclose(fid);
    fprintf('Report saved successfully.\n');
end
