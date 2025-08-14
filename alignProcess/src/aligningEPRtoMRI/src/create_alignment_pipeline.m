function create_alignment_pipeline()
    % Create a comprehensive EPR-to-MRI alignment pipeline
    % This script combines fiducial detection and alignment into one workflow
    
    fprintf('=== EPR-to-MRI Alignment Pipeline ===\n');
    fprintf('This pipeline will:\n');
    fprintf('1. Run fiducial detection on your data\n');
    fprintf('2. Extract fiducial coordinates from detected masks\n');
    fprintf('3. Calculate optimal alignment transformations\n');
    fprintf('4. Apply transformations and save aligned data\n\n');
    
    % Parameters
    input_file = input('Enter path to input .mat file: ', 's');
    if isempty(input_file)
        input_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat';
        fprintf('Using default: %s\n', input_file);
    end
    
    % Check if input file exists
    if ~exist(input_file, 'file')
        error('Input file does not exist: %s', input_file);
    end
    
    % Generate output paths
    [input_path, input_name, ~] = fileparts(input_file);
    fiducials_output = fullfile(input_path, [input_name '_aligned.mat']);
    
    fprintf('Input file: %s\n', input_file);
    fprintf('Output file: %s\n', fiducials_output);
    
    % Step 1: Run fiducial detection
    fprintf('\n=== Step 1: Running Fiducial Detection ===\n');
    
    % Build the run command for fiducials detection
    fiducials_script = 'C:\Users\ftmen\Documents\mrialign\alignProcess\src\creatingFiducials\run.bat';
    
    if ~exist(fiducials_script, 'file')
        error('Fiducials detection script not found: %s', fiducials_script);
    end
    
    % Run fiducials detection
    cmd = sprintf('"%s" "%s"', fiducials_script, input_file);
    fprintf('Running: %s\n', cmd);
    
    [status, output] = system(cmd);
    
    if status ~= 0
        error('Fiducials detection failed:\n%s', output);
    end
    
    fprintf('Fiducials detection completed successfully!\n');
    fprintf('Output:\n%s\n', output);
    
    % Step 2: Find the generated fiducials file
    fprintf('\n=== Step 2: Finding Generated Fiducials File ===\n');
    
    % Extract the output path from the detection script output
    lines = strsplit(output, '\n');
    fiducials_file = '';
    
    for i = 1:length(lines)
        line = strtrim(lines{i});
        if contains(line, '_with_boxes.mat')
            % Extract the full path
            start_idx = strfind(line, 'C:\');
            if ~isempty(start_idx)
                fiducials_file = line(start_idx:end);
                break;
            end
        end
    end
    
    if isempty(fiducials_file) || ~exist(fiducials_file, 'file')
        error('Could not find generated fiducials file');
    end
    
    fprintf('Found fiducials file: %s\n', fiducials_file);
    
    % Step 3: Run alignment
    fprintf('\n=== Step 3: Running Alignment ===\n');
    
    % Load fiducials data
    fid_data = load(fiducials_file);
    
    % Extract fiducials
    fiducials = extract_fiducials_from_masks(fid_data);
    
    % Display summary
    display_alignment_summary(fiducials);
    
    % Calculate alignments
    transformations = calculate_all_alignments(fiducials);
    
    % Apply transformations
    apply_alignment_transformations(fid_data, transformations, fiducials_output);
    
    fprintf('\n=== Alignment Pipeline Complete! ===\n');
    fprintf('Aligned data saved to: %s\n', fiducials_output);
    
    % Step 4: Generate verification report
    fprintf('\n=== Step 4: Generating Verification Report ===\n');
    generate_alignment_report(input_file, fiducials_output, transformations);
end

function fiducials = extract_fiducials_from_masks(data)
    % Extract fiducial coordinates from logical masks
    
    fiducials = struct();
    
    for i = 1:length(data.images)
        img = data.images{i};
        img_name = clean_image_name(img.Name);
        
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
                end
            end
        end
    end
end

function clean_name = clean_image_name(name)
    % Clean image name for struct field use
    clean_name = name;
    if startsWith(clean_name, '>')
        clean_name = clean_name(2:end);
    end
    clean_name = matlab.lang.makeValidName(clean_name);
end

function display_alignment_summary(fiducials)
    % Display alignment summary
    
    fprintf('\nFIDUCIALS DETECTED:\n');
    field_names = fieldnames(fiducials);
    
    for i = 1:length(field_names)
        name = field_names{i};
        points = fiducials.(name);
        fprintf('  %s: %d fiducials\n', name, size(points, 1));
    end
end

function transformations = calculate_all_alignments(fiducials)
    % Calculate alignment transformations for all EPR images to MRI
    
    transformations = {};
    reference = 'MRI';
    epr_images = {'PreHemo', 'Hemo', 'Post_Tranfus', 'Post_Transfus2'};
    
    fprintf('\nCALCULATING ALIGNMENTS:\n');
    
    for i = 1:length(epr_images)
        epr_name = epr_images{i};
        
        if isfield(fiducials, epr_name) && isfield(fiducials, reference)
            source_points = fiducials.(epr_name);
            target_points = fiducials.(reference);
            
            if size(source_points, 1) >= 3 && size(target_points, 1) >= 3
                fprintf('  Aligning %s to %s (%d fiducials)...\n', epr_name, reference, size(source_points, 1));
                
                transform = calculate_optimal_transformation(source_points, target_points);
                transformations{end+1} = struct('name', ['>' epr_name], 'matrix', transform);
                
                fprintf('    Success!\n');
            else
                fprintf('  Skipping %s (insufficient fiducials: %d)\n', epr_name, size(source_points, 1));
            end
        end
    end
end

function transform_matrix = calculate_optimal_transformation(source_points, target_points)
    % Calculate optimal rigid transformation using least squares
    
    n_pairs = min(size(source_points, 1), size(target_points, 1));
    src = source_points(1:n_pairs, :);
    tgt = target_points(1:n_pairs, :);
    
    % Calculate centroids
    src_centroid = mean(src, 1);
    tgt_centroid = mean(tgt, 1);
    
    % Center points
    src_centered = src - src_centroid;
    tgt_centered = tgt - tgt_centroid;
    
    % Calculate scaling
    src_scale = sqrt(sum(src_centered(:).^2) / n_pairs);
    tgt_scale = sqrt(sum(tgt_centered(:).^2) / n_pairs);
    scale_factor = tgt_scale / src_scale;
    
    % Apply scaling
    src_scaled = src_centered * scale_factor;
    
    % Calculate rotation using SVD
    H = src_scaled' * tgt_centered;
    [U, ~, V] = svd(H);
    R = V * U';
    
    % Ensure proper rotation
    if det(R) < 0
        V(:, 3) = -V(:, 3);
        R = V * U';
    end
    
    % Calculate translation
    t = tgt_centroid' - R * (src_centroid * scale_factor)';
    
    % Build transformation matrix
    transform_matrix = eye(4);
    transform_matrix(1:3, 1:3) = R * scale_factor;
    transform_matrix(1:3, 4) = t;
end

function apply_alignment_transformations(original_data, transformations, output_file)
    % Apply transformations and save results
    
    new_data = original_data;
    
    fprintf('\nAPPLYING TRANSFORMATIONS:\n');
    
    for i = 1:length(transformations)
        trans = transformations{i};
        img_name = trans.name;
        transform_matrix = trans.matrix;
        
        for j = 1:length(new_data.images)
            img = new_data.images{j};
            if strcmp(img.Name, img_name)
                new_data.images{j}.A = transform_matrix * img.A;
                fprintf('  Applied transformation to %s\n', img_name);
                break;
            end
        end
    end
    
    save(output_file, '-struct', 'new_data');
    fprintf('  Saved aligned data to: %s\n', output_file);
end

function generate_alignment_report(original_file, aligned_file, transformations)
    % Generate a detailed alignment report
    
    fprintf('\nALIGNMENT REPORT:\n');
    fprintf('Original file: %s\n', original_file);
    fprintf('Aligned file: %s\n', aligned_file);
    fprintf('Number of aligned images: %d\n', length(transformations));
    
    for i = 1:length(transformations)
        trans = transformations{i};
        T = trans.matrix;
        
        % Extract transformation components
        scale_x = norm(T(1:3, 1));
        scale_y = norm(T(1:3, 2));
        scale_z = norm(T(1:3, 3));
        translation = T(1:3, 4);
        
        fprintf('\n%s:\n', trans.name);
        fprintf('  Scale: [%.4f, %.4f, %.4f]\n', scale_x, scale_y, scale_z);
        fprintf('  Translation: [%.2f, %.2f, %.2f]\n', translation(1), translation(2), translation(3));
    end
    
    fprintf('\nAlignment pipeline completed successfully!\n');
end
