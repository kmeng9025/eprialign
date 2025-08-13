function save_ai_kidneys_matlab(original_mat_file, kidney_mask, bounding_boxes, output_filename)
% SAVE_AI_KIDNEYS_MATLAB - Save AI kidney detection results to Arbuz-compatible .mat file
% 
% Inputs:
%   original_mat_file - Path to original .mat file (string)
%   kidney_mask - 3D kidney segmentation mask (uint8)
%   bounding_boxes - Cell array of bounding box structures
%   output_filename - Output filename (optional)
%
% This function preserves the original Arbuz project structure while adding AI results

    fprintf('💾 Saving AI kidney results (MATLAB version)...\n');
    
    try
        % Load original data to preserve Arbuz project structure
        fprintf('   📂 Loading original project structure...\n');
        original_data = load(original_mat_file);
        
        % Create output filename if not provided
        if nargin < 4 || isempty(output_filename)
            [filepath, name, ~] = fileparts(original_mat_file);
            output_filename = fullfile(filepath, [name '_with_AI_kidneys.mat']);
        end
        
        fprintf('   ✅ Original project structure loaded\n');
        
        % Start with complete original data
        output_data = original_data;
        
        % Add AI kidney detection results (preserving Arbuz compatibility)
        output_data.ai_kidney_mask = uint8(kidney_mask);
        output_data.ai_detection_timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        
        % Model information (MATLAB-compatible)
        output_data.ai_model_type = 'UNet3D';
        output_data.ai_training_f1_score = 0.836;
        output_data.ai_training_iou = 0.718;
        
        % Process bounding boxes
        if ~isempty(bounding_boxes)
            num_kidneys = length(bounding_boxes);
            output_data.ai_num_kidneys_detected = num_kidneys;
            
            % Initialize arrays for bounding box data
            output_data.ai_kidney_ids = 1:num_kidneys;
            output_data.ai_kidney_sizes = zeros(1, num_kidneys);
            output_data.ai_bounds_min_y = zeros(1, num_kidneys);
            output_data.ai_bounds_max_y = zeros(1, num_kidneys);
            output_data.ai_bounds_min_x = zeros(1, num_kidneys);
            output_data.ai_bounds_max_x = zeros(1, num_kidneys);
            output_data.ai_bounds_min_z = zeros(1, num_kidneys);
            output_data.ai_bounds_max_z = zeros(1, num_kidneys);
            output_data.ai_center_y = zeros(1, num_kidneys);
            output_data.ai_center_x = zeros(1, num_kidneys);
            output_data.ai_center_z = zeros(1, num_kidneys);
            
            % Extract bounding box information
            for i = 1:num_kidneys
                box = bounding_boxes{i};
                output_data.ai_kidney_sizes(i) = box.size;
                output_data.ai_bounds_min_y(i) = box.bounds(1);
                output_data.ai_bounds_max_y(i) = box.bounds(2);
                output_data.ai_bounds_min_x(i) = box.bounds(3);
                output_data.ai_bounds_max_x(i) = box.bounds(4);
                output_data.ai_bounds_min_z(i) = box.bounds(5);
                output_data.ai_bounds_max_z(i) = box.bounds(6);
                output_data.ai_center_y(i) = box.center(1);
                output_data.ai_center_x(i) = box.center(2);
                output_data.ai_center_z(i) = box.center(3);
            end
        else
            output_data.ai_num_kidneys_detected = 0;
        end
        
        % Add summary statistics
        total_kidney_voxels = sum(kidney_mask(:));
        total_voxels = numel(kidney_mask);
        coverage_percent = (total_kidney_voxels / total_voxels) * 100;
        
        output_data.ai_total_kidney_voxels = total_kidney_voxels;
        output_data.ai_total_voxels = total_voxels;
        output_data.ai_coverage_percent = coverage_percent;
        
        % Save to .mat file with MATLAB format
        fprintf('   💾 Saving Arbuz-compatible .mat file...\n');
        save(output_filename, '-struct', 'output_data', '-v7.3');
        
        fprintf('✅ Arbuz-compatible AI kidney results saved!\n');
        fprintf('   File: %s\n', output_filename);
        fprintf('   Kidneys detected: %d\n', output_data.ai_num_kidneys_detected);
        fprintf('   Coverage: %.2f%%\n', coverage_percent);
        fprintf('   📋 Original Arbuz project structure preserved\n');
        
    catch ME
        fprintf('❌ Error saving .mat file: %s\n', ME.message);
        rethrow(ME);
    end
end
