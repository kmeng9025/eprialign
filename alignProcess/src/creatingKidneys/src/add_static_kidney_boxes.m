function add_static_kidney_boxes(input_file, output_file)
% ADD_STATIC_KIDNEY_BOXES - Add two static kidney box slaves to Arbuz project
%
% This function adds two simple kidney box slaves at typical kidney locations

    fprintf('🔄 Adding static kidney boxes to Arbuz project...\n');
    fprintf('   📂 Input: %s\n', input_file);
    fprintf('   💾 Output: %s\n', output_file);
    
    try
        % Load original Arbuz project
        fprintf('   📂 Loading original Arbuz project...\n');
        data = load(input_file);
        
        % Find the main MRI image to add kidneys to
        if isfield(data, 'images') && ~isempty(data.images)
            images = data.images;
            
            fprintf('   🖼️  Processing %d images...\n', size(images, 2));
            
            % Find the first image with 3D MRI data
            for i = 1:size(images, 2)
                try
                    current_image = images{1, i};
                    
                    if isfield(current_image, 'data') && ~isempty(current_image.data)
                        mri_data = current_image.data;
                        
                        % Check if it's 3D data
                        if ndims(mri_data) >= 3 && min(size(mri_data)) > 1
                            fprintf('     ✅ Found target image #%d: %s\n', i, mat2str(size(mri_data)));
                            
                            % Create static kidney box slaves
                            kidney_slaves = create_static_kidney_boxes(size(mri_data));
                            
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
                            
                            % Update the image in the data
                            data.images{1, i} = current_image;
                            
                            fprintf('     🎯 Added %d static kidney boxes to image #%d\n', length(kidney_slaves), i);
                            break; % Only add to the first suitable image
                        end
                    end
                catch ME
                    fprintf('     ⚠️  Could not process image #%d: %s\n', i, ME.message);
                    continue;
                end
            end
        end
        
        % Add metadata
        data.static_kidneys_added = sprintf('2 static kidney boxes added on %s', datestr(now));
        
        % Save final file
        fprintf('   💾 Saving Arbuz file with static kidney boxes...\n');
        save(output_file, '-struct', 'data', '-v7.3');
        
        % Display success
        file_info = dir(output_file);
        fprintf('✅ Static kidney boxes added successfully!\n');
        fprintf('   📁 File: %s\n', output_file);
        fprintf('   📊 Size: %.1f MB\n', file_info.bytes / (1024*1024));
        fprintf('   🎯 Kidney boxes: 2 (left and right)\n');
        fprintf('📋 Ready for ArbuzGUI!\n');
        
    catch ME
        fprintf('❌ Error adding kidney boxes: %s\n', ME.message);
        rethrow(ME);
    end
end

function kidney_slaves = create_static_kidney_boxes(image_size)
% CREATE_STATIC_KIDNEY_BOXES - Create two static kidney box slaves
    
    fprintf('     🎨 Creating static kidney boxes for image size: %s\n', mat2str(image_size));
    
    width = image_size(1);
    height = image_size(2);
    depth = image_size(3);
    
    % Define kidney box parameters
    box_width = round(width * 0.15);   % 15% of image width
    box_height = round(height * 0.15); % 15% of image height
    box_depth = round(depth * 0.4);    % 40% of image depth
    
    % Left kidney position (patient's left, image right)
    left_center_x = round(width * 0.7);
    left_center_y = round(height * 0.4);
    left_center_z = round(depth * 0.5);
    
    % Right kidney position (patient's right, image left)
    right_center_x = round(width * 0.3);
    right_center_y = round(height * 0.4);
    right_center_z = round(depth * 0.5);
    
    kidney_slaves = {};
    
    % Create left kidney box
    left_kidney_mask = false(image_size);
    left_x_start = max(1, left_center_x - round(box_width/2));
    left_x_end = min(width, left_center_x + round(box_width/2));
    left_y_start = max(1, left_center_y - round(box_height/2));
    left_y_end = min(height, left_center_y + round(box_height/2));
    left_z_start = max(1, left_center_z - round(box_depth/2));
    left_z_end = min(depth, left_center_z + round(box_depth/2));
    
    left_kidney_mask(left_x_start:left_x_end, left_y_start:left_y_end, left_z_start:left_z_end) = true;
    
    % Create left kidney slave
    left_slave = struct();
    left_slave.Name = 'Left Kidney Box';
    left_slave.ImageType = '3DMASK';
    left_slave.data = logical(left_kidney_mask);
    left_slave.FileName = '';
    left_slave.Selected = 0;
    left_slave.Visible = 0;
    left_slave.isLoaded = 0;
    left_slave.isStore = 1;
    left_slave.A = eye(4);
    left_slave.box = image_size;
    left_slave.Anative = eye(4);
    left_slave.Color = [1, 0, 0];  % Red
    
    kidney_slaves{end+1} = left_slave;
    
    % Create right kidney box
    right_kidney_mask = false(image_size);
    right_x_start = max(1, right_center_x - round(box_width/2));
    right_x_end = min(width, right_center_x + round(box_width/2));
    right_y_start = max(1, right_center_y - round(box_height/2));
    right_y_end = min(height, right_center_y + round(box_height/2));
    right_z_start = max(1, right_center_z - round(box_depth/2));
    right_z_end = min(depth, right_center_z + round(box_depth/2));
    
    right_kidney_mask(right_x_start:right_x_end, right_y_start:right_y_end, right_z_start:right_z_end) = true;
    
    % Create right kidney slave
    right_slave = struct();
    right_slave.Name = 'Right Kidney Box';
    right_slave.ImageType = '3DMASK';
    right_slave.data = logical(right_kidney_mask);
    right_slave.FileName = '';
    right_slave.Selected = 0;
    right_slave.Visible = 0;
    right_slave.isLoaded = 0;
    right_slave.isStore = 1;
    right_slave.A = eye(4);
    right_slave.box = image_size;
    right_slave.Anative = eye(4);
    right_slave.Color = [0, 1, 0];  % Green
    
    kidney_slaves{end+1} = right_slave;
    
    left_size = sum(left_kidney_mask(:));
    right_size = sum(right_kidney_mask(:));
    
    fprintf('       🟥 Left kidney box: %d voxels at (%d,%d,%d)\n', left_size, left_center_x, left_center_y, left_center_z);
    fprintf('       🟩 Right kidney box: %d voxels at (%d,%d,%d)\n', right_size, right_center_x, right_center_y, right_center_z);
    fprintf('     ✅ Created 2 static kidney box slaves\n');
end
