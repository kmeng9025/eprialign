function create_clean_arbuz_with_kidney_masks(original_file, ai_results_file, output_file)
% CREATE_CLEAN_ARBUZ_WITH_KIDNEY_MASKS - Create final .mat with masks drawn on images
%
% This function:
% 1. Loads original Arbuz project
% 2. Loads AI kidney results 
% 3. Draws kidney masks directly onto the MRI images
% 4. Saves clean final .mat file
% 5. Cleans up temporary files

    fprintf('🔄 Creating final Arbuz file with kidney masks drawn on images...\n');
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
        
        % Get kidney mask
        kidney_mask = ai_data.ai_kidney_mask;
        num_kidneys = ai_data.ai_num_kidneys_detected;
        
        fprintf('   🎯 Processing %d detected kidneys...\n', num_kidneys);
        
        % Start with original data
        output_data = original_data;
        
        % Find and modify the MRI images with kidney overlays
        fprintf('   🖼️  Drawing kidney masks onto MRI images...\n');
        
        if isfield(output_data, 'images') && ~isempty(output_data.images)
            images = output_data.images;
            
            % Find the image with MRI data
            for i = 1:size(images, 2)
                try
                    img_struct = images(1, i);
                    
                    % Check if this image has data
                    if isfield(img_struct, 'data') && ~isempty(img_struct.data)
                        mri_data = img_struct.data;
                        
                        % Check if it's 3D data
                        if ndims(mri_data) >= 3 && min(size(mri_data)) > 1
                            fprintf('     ✅ Found MRI data in images(1,%d): %s\n', i, mat2str(size(mri_data)));
                            
                            % Draw kidney masks onto the MRI data
                            modified_mri = draw_kidney_masks_on_mri(mri_data, kidney_mask, num_kidneys);
                            
                            % Update the image data
                            output_data.images(1, i).data = modified_mri;
                            
                            fprintf('     🎨 Kidney masks drawn onto MRI images\n');
                            break;
                        end
                    end
                catch ME
                    fprintf('     ⚠️  Could not process images(1,%d): %s\n', i, ME.message);
                    continue;
                end
            end
        end
        
        % Add minimal AI metadata (no redundant data)
        output_data.ai_detection_summary = sprintf('%d kidneys detected on %s', ...
            num_kidneys, ai_data.ai_detection_timestamp);
        output_data.ai_model_info = sprintf('UNet3D (F1=%.3f)', ai_data.ai_training_f1_score);
        
        % Save final clean file
        fprintf('   💾 Saving final Arbuz file with kidney masks...\n');
        save(output_file, '-struct', 'output_data', '-v7.3');
        
        % Clean up temporary files in the same directory
        [output_dir, ~, ~] = fileparts(output_file);
        cleanup_temp_files(output_dir);
        
        % Display success
        file_info = dir(output_file);
        fprintf('✅ Final Arbuz file created successfully!\n');
        fprintf('   📁 File: %s\n', output_file);
        fprintf('   📊 Size: %.1f MB\n', file_info.bytes / (1024*1024));
        fprintf('   🎯 Kidneys: %d (masks drawn on images)\n', num_kidneys);
        fprintf('   🧹 Temporary files cleaned up\n');
        fprintf('📋 Ready for ArbuzGUI!\n');
        
    catch ME
        fprintf('❌ Error creating final file: %s\n', ME.message);
        rethrow(ME);
    end
end

function modified_mri = draw_kidney_masks_on_mri(mri_data, kidney_mask, num_kidneys)
% DRAW_KIDNEY_MASKS_ON_MRI - Draw kidney masks directly onto MRI images
    
    fprintf('     🎨 Drawing kidney masks onto MRI...\n');
    
    % Start with original MRI data
    modified_mri = double(mri_data);
    
    % Get data range for scaling
    mri_min = min(modified_mri(:));
    mri_max = max(modified_mri(:));
    mri_range = mri_max - mri_min;
    
    % Create kidney overlay with different intensities
    kidney_overlay = zeros(size(modified_mri));
    
    if num_kidneys > 0
        % Label connected components to identify individual kidneys
        labeled_mask = bwlabeln(kidney_mask);
        
        % Color each kidney differently using intensity values
        for k = 1:num_kidneys
            kidney_pixels = (labeled_mask == k);
            
            if sum(kidney_pixels(:)) > 0
                % Use high intensity for kidney overlay
                intensity_boost = mri_range * 0.3; % 30% brightness boost
                kidney_overlay(kidney_pixels) = intensity_boost;
                
                fprintf('       🟥 Kidney %d: %d pixels enhanced\n', k, sum(kidney_pixels(:)));
            end
        end
        
        % Apply the overlay to the MRI data
        modified_mri = modified_mri + kidney_overlay;
        
        % Ensure we don't exceed original data type range
        modified_mri = min(modified_mri, mri_max * 1.3); % Allow some overflow for visibility
        
        fprintf('     ✅ Kidney masks integrated into MRI data\n');
    else
        fprintf('     ⚠️  No kidneys to draw\n');
    end
    
    % Convert back to original data type
    if isa(mri_data, 'uint16')
        modified_mri = uint16(modified_mri);
    elseif isa(mri_data, 'uint8')
        modified_mri = uint8(modified_mri);
    elseif isa(mri_data, 'single')
        modified_mri = single(modified_mri);
    end
end

function cleanup_temp_files(output_dir)
% CLEANUP_TEMP_FILES - Remove temporary files, keep only final result
    
    fprintf('   🧹 Cleaning up temporary files...\n');
    
    try
        % List of temporary file patterns to remove
        temp_patterns = {
            '*_AI_results.mat',
            '*_with_AI_kidneys.mat', 
            '*_ARBUZ_COMPATIBLE.mat',
            '*.png',
            '*.jpg',
            '*_summary.*',
            '*ENHANCED*.*'
        };
        
        files_removed = 0;
        
        for i = 1:length(temp_patterns)
            pattern = temp_patterns{i};
            temp_files = dir(fullfile(output_dir, pattern));
            
            for j = 1:length(temp_files)
                temp_file_path = fullfile(output_dir, temp_files(j).name);
                
                % Don't delete the final output file
                if ~contains(temp_files(j).name, 'FINAL_CLEAN')
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
