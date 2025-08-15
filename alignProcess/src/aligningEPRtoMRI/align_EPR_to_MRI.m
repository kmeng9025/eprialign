function align_EPR_to_MRI()
    % Main function to align EPR images to MRI using fiducials
    % This implements a complete pipeline for fiducial-based registration
    
    fprintf('Starting EPR-to-MRI alignment using fiducials...\n');
    
    % File paths
    fiducials_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\fiducials_withoutROIwithMRIwithoutTransformations_20250814_125342\box_fiducials_withoutROIwithMRIwithoutTransformations_20250814_125345\withoutROIwithMRIwithoutTransformations_with_boxes.mat';
    output_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI_aligned.mat';
    
    % Load fiducials data
    fprintf('Loading fiducials data...\n');
    fid_data = load(fiducials_file);
    
    % Extract fiducials for each image
    fiducials = extract_fiducials_from_data(fid_data);
    
    % Display fiducials summary
    display_fiducials_summary(fiducials);
    
    % Calculate alignment transformations
    fprintf('\nCalculating alignment transformations...\n');
    
    % Use MRI as reference (target)
    reference_image = '>MRI';
    
    % Find EPR images to align
    epr_images = {'PreHemo', 'Hemo', 'Post_Transfus', 'Post_Transfus2'};
    
    % Calculate transformations for each EPR image
    transformations = {};
    for i = 1:length(epr_images)
        epr_name = epr_images{i};
        if isfield(fiducials, epr_name) && isfield(fiducials, 'MRI')
            fprintf('  Aligning %s to %s...\n', epr_name, 'MRI');
            
            % Get fiducial points
            source_points = fiducials.(epr_name);
            target_points = fiducials.MRI;
            
            % Calculate transformation
            if size(source_points, 1) >= 3 && size(target_points, 1) >= 3
                transform = calculate_fiducial_alignment(source_points, target_points);
                transformations{end+1} = struct('name', ['>' epr_name], 'matrix', transform, 'target', '>MRI');
                
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
    
    fprintf('Alignment complete! Output saved to: %s\n', output_file);
end

function fiducials = extract_fiducials_from_data(data)
    % Extract fiducial coordinates from the logical masks in slaves data
    
    fiducials = struct();
    
    % Process each image
    for i = 1:length(data.images)
        img = data.images{i};
        img_name = clean_image_name(img.Name);
        
        % Look for slaves (fiducials)
        if isfield(img, 'slaves') && ~isempty(img.slaves)
            slave = img.slaves{1}; % Should be only one slave per image
            
            if isfield(slave, 'data') && ~isempty(slave.data)
                % slave.data is a logical mask
                mask = slave.data;
                
                % Find connected components (individual fiducials)
                cc = bwconncomp(mask);
                
                if cc.NumObjects > 0
                    points = zeros(cc.NumObjects, 3);
                    
                    for j = 1:cc.NumObjects
                        % Get indices of this connected component
                        [y, x, z] = ind2sub(size(mask), cc.PixelIdxList{j});
                        
                        % Calculate centroid
                        centroid_y = mean(y);
                        centroid_x = mean(x);
                        centroid_z = mean(z);
                        
                        points(j, :) = [centroid_x, centroid_y, centroid_z];
                    end
                    
                    fiducials.(img_name) = points;
                    fprintf('  Found %d fiducials for %s\n', size(points, 1), img_name);
                else
                    fprintf('  No fiducials detected for %s\n', img_name);
                end
            else
                fprintf('  No data field in slave for %s\n', img_name);
            end
        else
            fprintf('  No slaves found for %s\n', img_name);
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

function display_fiducials_summary(fiducials)
    % Display summary of extracted fiducials
    
    fprintf('\nFIDUCIALS SUMMARY:\n');
    field_names = fieldnames(fiducials);
    
    for i = 1:length(field_names)
        name = field_names{i};
        points = fiducials.(name);
        
        fprintf('  %s: %d fiducials\n', name, size(points, 1));
        
        % Show first few points
        if size(points, 1) > 0
            fprintf('    Example points:\n');
            for j = 1:min(3, size(points, 1))
                fprintf('      [%.2f, %.2f, %.2f]\n', points(j, 1), points(j, 2), points(j, 3));
            end
        end
    end
end

function transform_matrix = calculate_fiducial_alignment(source_points, target_points)
    % Calculate 4x4 transformation matrix to align source points to target points
    % Uses least squares approach to find optimal rigid transformation
    
    % Number of point pairs
    n_pairs = min(size(source_points, 1), size(target_points, 1));
    
    if n_pairs < 3
        error('Need at least 3 point pairs for alignment');
    end
    
    % Use only the matching number of points
    src = source_points(1:n_pairs, :);
    tgt = target_points(1:n_pairs, :);
    
    fprintf('      Using %d point pairs for alignment\n', n_pairs);
    
    % Calculate centroids
    src_centroid = mean(src, 1);
    tgt_centroid = mean(tgt, 1);
    
    % Center the points
    src_centered = src - src_centroid;
    tgt_centered = tgt - tgt_centroid;
    
    % Calculate scaling factor
    src_scale = sqrt(sum(src_centered(:).^2) / n_pairs);
    tgt_scale = sqrt(sum(tgt_centered(:).^2) / n_pairs);
    scale_factor = tgt_scale / src_scale;
    
    fprintf('      Scale factor: %.4f\n', scale_factor);
    
    % Apply scaling to source points
    src_scaled = src_centered * scale_factor;
    
    % Calculate optimal rotation using SVD
    H = src_scaled' * tgt_centered;
    [U, ~, V] = svd(H);
    R = V * U';
    
    % Ensure proper rotation (det(R) = 1)
    if det(R) < 0
        V(:, 3) = -V(:, 3);
        R = V * U';
    end
    
    % Calculate translation
    t = tgt_centroid' - R * (src_centroid * scale_factor)';
    
    % Construct 4x4 transformation matrix
    transform_matrix = eye(4);
    transform_matrix(1:3, 1:3) = R * scale_factor;
    transform_matrix(1:3, 4) = t;
    
    % Calculate alignment error
    src_transformed = (src * scale_factor * R' + t')';
    error_distances = sqrt(sum((src_transformed' - tgt).^2, 2));
    mean_error = mean(error_distances);
    max_error = max(error_distances);
    
    fprintf('      Mean alignment error: %.4f\n', mean_error);
    fprintf('      Max alignment error: %.4f\n', max_error);
end

function apply_transformations_to_data(original_data, transformations, output_file)
    % Apply calculated transformations to the data and save
    
    % Copy original data
    new_data = original_data;
    
    % Apply transformations to each image
    for i = 1:length(transformations)
        trans = transformations{i};
        img_name = trans.name;
        transform_matrix = trans.matrix;
        
        % Find the image in the data
        for j = 1:length(new_data.images)
            img = new_data.images{j};
            if strcmp(img.Name, img_name)
                % Apply transformation
                fprintf('  Applying transformation to %s\n', img_name);
                
                % Update the A matrix (current transformation)
                new_data.images{j}.A = transform_matrix * img.A;
                
                fprintf('    Updated transformation matrix for %s\n', img_name);
                break;
            end
        end
    end
    
    % Save the result
    fprintf('  Saving aligned data to: %s\n', output_file);
    save(output_file, '-struct', 'new_data');
end
