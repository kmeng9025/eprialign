function add_test_square_mask(input_file, output_file)
% ADD_TEST_SQUARE_MASK - Add a simple test square mask to verify pipeline
%
% This function adds a static square mask in the center to test the workflow

    fprintf('🔲 Adding test square mask to Arbuz project...\n');
    fprintf('   📂 Input: %s\n', input_file);
    fprintf('   💾 Output: %s\n', output_file);
    
    try
        % Load original Arbuz project
        fprintf('   📂 Loading original Arbuz project...\n');
        data = load(input_file);
        
        % Find the main MRI image to add test mask to
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
                            
                            % Create test square mask
                            test_mask_slave = create_test_square_mask(size(mri_data));
                            
                            % Add test mask to the image
                            if isfield(current_image, 'slaves') && ~isempty(current_image.slaves)
                                % Append to existing slaves
                                current_image.slaves{end+1} = test_mask_slave;
                            else
                                % Create new slaves cell array
                                current_image.slaves = {test_mask_slave};
                            end
                            
                            % Update the image in the data
                            data.images{1, i} = current_image;
                            
                            fprintf('     🎯 Added test square mask to image #%d\n', i);
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
        data.test_mask_added = sprintf('Test square mask added on %s', datestr(now));
        
        % Save final file
        fprintf('   💾 Saving Arbuz file with test mask...\n');
        save(output_file, '-struct', 'data', '-v7.3');
        
        % Display success
        file_info = dir(output_file);
        fprintf('✅ Test square mask added successfully!\n');
        fprintf('   📁 File: %s\n', output_file);
        fprintf('   📊 Size: %.1f MB\n', file_info.bytes / (1024*1024));
        fprintf('   🔲 Test mask: 1 square in center\n');
        fprintf('📋 Ready for ArbuzGUI!\n');
        
    catch ME
        fprintf('❌ Error adding test mask: %s\n', ME.message);
        rethrow(ME);
    end
end

function test_slave = create_test_square_mask(image_size)
    % CREATE_TEST_SQUARE_MASK - Create a simple square mask in the center
    %
    % Input:
    %   image_size - [height, width, depth] of the image
    %
    % Output:
    %   test_slave - Slave structure with square mask
    
    fprintf('   🎨 Creating test square mask for image size: %s\n', mat2str(image_size));
    
    height = image_size(1);
    width = image_size(2);
    depth = image_size(3);
    
    % Calculate square dimensions (about 15% of image size)
    square_size = round(min(height, width) * 0.15);
    square_depth = max(1, round(depth * 0.2));
    
    % Calculate center position
    center_x = round(width / 2);
    center_y = round(height / 2);
    center_z = round(depth / 2);
    
    % Create empty mask
    test_mask = false(image_size);
    
    % Define square boundaries
    x_start = max(1, center_x - round(square_size/2));
    x_end = min(width, center_x + round(square_size/2));
    y_start = max(1, center_y - round(square_size/2));
    y_end = min(height, center_y + round(square_size/2));
    z_start = max(1, center_z - round(square_depth/2));
    z_end = min(depth, center_z + round(square_depth/2));
    
    % Fill the square region
    test_mask(y_start:y_end, x_start:x_end, z_start:z_end) = true;
    
    % Create test slave structure
    test_slave = struct();
    test_slave.Name = 'TestSquare';
    test_slave.ImageType = '3DMASK';
    test_slave.data = logical(test_mask);
    test_slave.FileName = '';
    test_slave.Selected = 0;
    test_slave.Visible = 0;
    test_slave.isLoaded = 0;
    test_slave.isStore = 1;
    test_slave.A = eye(4);
    test_slave.box = image_size;
    test_slave.Anative = eye(4);
    test_slave.Color = [1, 0, 1];  % Magenta
    
    mask_size = sum(test_mask(:));
    coverage = mask_size / prod(image_size) * 100;
    
    fprintf('     🔲 Test square: %d voxels at (%d,%d,%d), %.2f%% coverage\n', ...
            mask_size, center_x, center_y, center_z, coverage);
    fprintf('     ✅ Created test square mask slave\n');
end
