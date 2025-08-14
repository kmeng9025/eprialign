function create_kidney_slaves_final(original_file, ai_results_file, output_file)
% CREATE_KIDNEY_SLAVES_FINAL - Create kidney slaves in Arbuz project
%
% This function:
% 1. Loads original Arbuz project
% 2. Loads AI kidney results 
% 3. Creates kidney slaves using the same structure as fiducials
% 4. Saves final .mat file with kidney slaves
% 5. Cleans up temporary files

    fprintf('🔄 Creating Arbuz project with kidney slaves...\n');
    fprintf('   📂 Original: %s\n', original_file);
    fprintf('   🤖 AI results: %s\n', ai_results_file);
    fprintf('   💾 Output: %s\n', output_file);
    
    try
        % Load original Arbuz project
        fprintf('   📂 Loading original Arbuz project...\n');
        original_data = load(original_file);
        
        % Load AI results
        fprintf('   🤖 Loading AI kidney results...\n');
        ai_data = load(ai_results_file);
        
        % Check if we have multiple MRI results or single result
        if isfield(ai_data, 'ai_kidney_masks')
            % Multiple MRI images processed
            kidney_masks = ai_data.ai_kidney_masks;
            results_summary = ai_data.ai_results_summary;
            total_kidneys = ai_data.ai_total_kidneys_detected;
            num_mri_processed = ai_data.ai_num_mri_images_processed;
            
            fprintf('   🎯 Processing %d total kidneys from %d MRI images...\n', total_kidneys, num_mri_processed);
        else
            % Single MRI image (backward compatibility)
            kidney_mask = ai_data.ai_kidney_mask;
            num_kidneys = ai_data.ai_num_kidneys_detected;
            confidence = ai_data.ai_detection_confidence;
            
            fprintf('   🎯 Processing %d detected kidneys (confidence: %.3f)...\n', num_kidneys, confidence);
        end
        
        % Start with original data
        output_data = original_data;
        
        % Add kidney slaves to images
        if isfield(output_data, 'images') && ~isempty(output_data.images)
            images = output_data.images;
            
            fprintf('   🖼️  Processing %d images...\n', size(images, 2));
            
            % Process each image
            for i = 1:size(images, 2)
                try
                    current_image = images{1, i};
                    
                    % Check if this image has MRI data
                    if isfield(current_image, 'data') && ~isempty(current_image.data)
                        mri_data = current_image.data;
                        
                        % Check if it's 3D data
                        if ndims(mri_data) >= 3 && min(size(mri_data)) > 1
                            % Get image name for matching with AI results
                            image_name = '';
                            if isfield(current_image, 'Name') && ~isempty(current_image.Name)
                                if ischar(current_image.Name)
                                    image_name = current_image.Name;
                                else
                                    image_name = char(current_image.Name);
                                end
                            end
                            
                            fprintf('     📂 Checking image #%d: %s (%s)\n', i, mat2str(size(mri_data)), image_name);
                            
                            % Find matching kidney mask for this image
                            kidney_mask_for_image = [];
                            num_kidneys_for_image = 0;
                            confidence_for_image = 0;
                            
                            if exist('kidney_masks', 'var')
                                % Multiple MRI case - find matching mask
                                mask_names = fieldnames(kidney_masks);
                                
                                for mask_idx = 1:length(mask_names)
                                    mask_name = mask_names{mask_idx};
                                    
                                    % Find corresponding result first to get exact dimensions
                                    matching_result = [];
                                    for res_idx = 1:length(results_summary)
                                        result = results_summary{res_idx};
                                        if strcmp(result.image_name, mask_name)
                                            matching_result = result;
                                            break;
                                        end
                                    end
                                    
                                    % Match by exact name first, then by dimensions
                                    exact_name_match = ~isempty(image_name) && strcmp(image_name, mask_name);
                                    
                                    % Match by dimensions - check that the mask dimensions match the image dimensions
                                    dimension_match = false;
                                    if ~isempty(matching_result)
                                        mask_data = kidney_masks.(mask_name);
                                        if isequal(size(mri_data), size(mask_data))
                                            dimension_match = true;
                                        end
                                    end
                                    
                                    % Use exact name match first, then dimension match as fallback
                                    if exact_name_match || dimension_match
                                        
                                        % Find corresponding result
                                        for res_idx = 1:length(results_summary)
                                            result = results_summary{res_idx};
                                            if strcmp(result.image_name, mask_name)
                                                
                                                kidney_mask_for_image = kidney_masks.(mask_name);
                                                num_kidneys_for_image = result.num_kidneys;
                                                confidence_for_image = result.confidence;
                                                
                                                fprintf('     ✅ Found kidney data for %s: %d kidneys (dimensions: %s)\n', mask_name, num_kidneys_for_image, mat2str(size(kidney_mask_for_image)));
                                                break;
                                            end
                                        end
                                        
                                        if ~isempty(kidney_mask_for_image)
                                            break;
                                        end
                                    end
                                end
                            else
                                % Single MRI case (backward compatibility)
                                if size(mri_data) == size(kidney_mask)
                                    kidney_mask_for_image = kidney_mask;
                                    num_kidneys_for_image = num_kidneys;
                                    confidence_for_image = confidence;
                                    
                                    fprintf('     ✅ Found target image #%d: %s\n', i, mat2str(size(mri_data)));
                                end
                            end
                            
                            % Create kidney slaves if we found matching data
                            if ~isempty(kidney_mask_for_image) && num_kidneys_for_image > 0
                                kidney_slaves = create_individual_kidney_slaves(kidney_mask_for_image, num_kidneys_for_image, current_image);
                                
                                if ~isempty(kidney_slaves)
                                    % Add kidney slaves to the image
                                    if isfield(current_image, 'slaves') && ~isempty(current_image.slaves)
                                        % Append to existing slaves
                                        for k = 1:length(kidney_slaves)
                                            current_image.slaves{end+1} = kidney_slaves{k};
                                        end
                                    else
                                        % Create new slaves cell array
                                        current_image.slaves = kidney_slaves;
                                    end
                                    
                                    % Update the image in the output
                                    output_data.images{1, i} = current_image;
                                    
                                    fprintf('     🎯 Added %d kidney slaves to image #%d\n', length(kidney_slaves), i);
                                end
                            end
                        end
                    end
                catch ME
                    fprintf('     ⚠️  Could not process image #%d: %s\n', i, ME.message);
                    continue;
                end
            end
        end
        
        % Add minimal AI metadata
        if exist('total_kidneys', 'var')
            % Multiple MRI case
            output_data.ai_detection_summary = sprintf('%d kidney slaves created from %d MRI images on %s', ...
                total_kidneys, num_mri_processed, ai_data.ai_detection_timestamp);
            output_data.ai_model_info = sprintf('UNet3D (F1=%.3f) processed %d MRI images', ...
                ai_data.ai_training_f1_score, num_mri_processed);
            total_kidneys_for_display = total_kidneys;
        else
            % Single MRI case
            output_data.ai_detection_summary = sprintf('%d kidney slaves created on %s', ...
                num_kidneys, ai_data.ai_detection_timestamp);
            output_data.ai_model_info = sprintf('UNet3D (F1=%.3f, confidence=%.3f)', ...
                ai_data.ai_training_f1_score, confidence);
            total_kidneys_for_display = num_kidneys;
        end
        
        % Save final clean file
        fprintf('   💾 Saving final Arbuz file with kidney slaves...\n');
        save(output_file, '-struct', 'output_data', '-v7.3');
        
        % Clean up temporary files in the same directory
        [output_dir, ~, ~] = fileparts(output_file);
        cleanup_temp_files(output_dir);
        
        % Display success
        file_info = dir(output_file);
        fprintf('✅ Final Arbuz file with kidney slaves created!\n');
        fprintf('   📁 File: %s\n', output_file);
        fprintf('   📊 Size: %d bytes\n', file_info.bytes);
        fprintf('   🎯 Kidney slaves: %d\n', total_kidneys_for_display);
        fprintf('   🧹 Temporary files cleaned up\n');
        fprintf('📋 Ready for ArbuzGUI - kidneys will appear as slaves!\n');
        
    catch ME
        fprintf('❌ Error creating kidney slaves: %s\n', ME.message);
        rethrow(ME);
    end
end

function kidney_slaves = create_individual_kidney_slaves(kidney_mask, num_kidneys, parent_image)
% CREATE_INDIVIDUAL_KIDNEY_SLAVES - Create individual slave for each kidney
    
    kidney_slaves = {};
    
    if num_kidneys == 0
        fprintf('     ⚠️  No kidneys to create slaves for\n');
        return;
    end
    
    % Label connected components to separate individual kidneys
    labeled_mask = bwlabeln(kidney_mask);
    
    % Create a slave for each kidney
    kidney_count = 0;
    for k = 1:max(labeled_mask(:))
        % Extract individual kidney mask
        individual_kidney = (labeled_mask == k);
        
        % Check if kidney is large enough
        kidney_size = sum(individual_kidney(:));
        if kidney_size < 100  % Minimum size threshold
            continue;
        end
        
        kidney_count = kidney_count + 1;
        
        % Create kidney slave using the same structure as working training files
        kidney_slave = struct();
        kidney_slave.Name = sprintf('AI Kidney %d', kidney_count);
        kidney_slave.ImageType = '3DMASK';
        kidney_slave.data = logical(individual_kidney);  % Convert to logical
        kidney_slave.FileName = '';
        kidney_slave.Selected = 0;
        kidney_slave.Visible = 1;  % Make kidney visible by default
        kidney_slave.isLoaded = 1;  % Mark kidney data as loaded
        kidney_slave.isStore = 1;   % Store in project
        kidney_slave.A = eye(4);
        kidney_slave.box = size(individual_kidney);
        kidney_slave.Color = [1, 0, 0];  % Red color for kidneys
        
        % Copy Anative from parent
        if isfield(parent_image, 'Anative')
            kidney_slave.Anative = parent_image.Anative;
        else
            kidney_slave.Anative = eye(4);
        end
        
        % Add to kidney slaves list
        kidney_slaves{end+1} = kidney_slave;
        
        fprintf('     🟥 Created kidney slave %d: %d voxels\n', kidney_count, kidney_size);
    end
    
    fprintf('     ✅ Total kidney slaves created: %d\n', length(kidney_slaves));
end

function cleanup_temp_files(output_dir)
% CLEANUP_TEMP_FILES - Remove temporary files, keep only final result
    
    fprintf('   🧹 Cleaning up temporary files...\n');
    
    try
        % List of temporary file patterns to remove
        temp_patterns = {
            '*_AI_results.mat',
            '*_ARBUZ_COMPATIBLE.mat',
            '*.png',
            '*.jpg',
            '*_summary.*',
            '*ENHANCED*.*',
            'temp_*.mat'
        };
        
        files_removed = 0;
        
        for i = 1:length(temp_patterns)
            pattern = temp_patterns{i};
            temp_files = dir(fullfile(output_dir, pattern));
            
            for j = 1:length(temp_files)
                temp_file_path = fullfile(output_dir, temp_files(j).name);
                
                % Don't delete the final output file
                if ~contains(temp_files(j).name, 'FINAL_KIDNEY_SLAVES')
                    try
                        delete(temp_file_path);
                        files_removed = files_removed + 1;
                        fprintf('     🗑️  Removed: %s\n', temp_files(j).name);
                    catch
                        fprintf('     ⚠️  Could not remove: %s\n', temp_files(j).name);
                    end
                end
            end
        end
        
        fprintf('   ✅ Cleaned up %d temporary files\n', files_removed);
        
    catch ME
        fprintf('   ⚠️  Cleanup warning: %s\n', ME.message);
    end
end
