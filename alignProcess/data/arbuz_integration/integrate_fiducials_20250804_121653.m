% MATLAB script to integrate AI-detected fiducials into ArbuzGUI project
% Generated automatically by Python fiducial detection system

function integrate_fiducials_20250804_121653()
    fprintf('>> Integrating AI-detected fiducials into ArbuzGUI project\n');
    fprintf('================================================================\n');
    
    % Add ArbuzGUI to path (adjust path as needed)
    arbuz_path = fullfile(pwd, '..', '..', 'Arbuz2.0');
    if exist(arbuz_path, 'dir')
        addpath(arbuz_path);
        addpath(fullfile(arbuz_path, 'Routines'));
        fprintf('>> Added ArbuzGUI to path: %s\n', arbuz_path);
    else
        fprintf('!! ArbuzGUI directory not found: %s\n', arbuz_path);
        fprintf('   Please adjust the arbuz_path variable in this script\n');
        return;
    end
    
    % Load fiducial data
    fiducial_data_file = '../data/arbuz_integration/fiducial_data_20250804_121653.mat';
    fprintf('>> Loading fiducial data: %s\n', fiducial_data_file);
    
    try
        fid_data = load(fiducial_data_file);
        fprintf('>> Loaded fiducial data for %d images\n', fid_data.processed_images);
    catch ME
        fprintf('!! Error loading fiducial data: %s\n', ME.message);
        return;
    end
    
    % Initialize ArbuzGUI
    try
        hGUI = figure('Visible', 'off', 'Name', 'ArbuzGUI Fiducial Integration');
        arbuz_InitializeProject(hGUI);
        fprintf('>> Initialized ArbuzGUI\n');
    catch ME
        fprintf('!! Error initializing ArbuzGUI: %s\n', ME.message);
        return;
    end
    
    % Load the original project
    project_file = '../data/training/withoutROI.mat';
    fprintf('>> Opening original project: %s\n', project_file);
    
    try
        status = arbuz_OpenProject(hGUI, project_file);
        if status
            fprintf('>> Project opened successfully\n');
        else
            fprintf('!! Failed to open project\n');
            return;
        end
    catch ME
        fprintf('!! Error opening project: %s\n', ME.message);
        return;
    end
    
    % Add fiducials to each image
    image_names = fieldnames(fid_data.fiducial_results);
    added_count = 0;
    
    for i = 1:length(image_names)
        image_name = image_names{i};
        fid_result = fid_data.fiducial_results.(image_name);
        
        fprintf('\n>> Adding fiducials to image: %s\n', image_name);
        fprintf('   Fiducials detected: %d\n', fid_result.num_fiducials);
        
        if fid_result.num_fiducials > 0
            % Create slave image structure
            slave_image = struct();
            slave_image.Name = 'Fiducials_AI';
            slave_image.ImageType = 'MASK';
            slave_image.data = fid_result.mask;
            slave_image.FileName = '';
            slave_image.Selected = 0;
            slave_image.Visible = 1;
            slave_image.isLoaded = 1;
            slave_image.isStore = 1;
            
            % Add slave to the master image
            try
                status = arbuz_AddImage(hGUI, slave_image, image_name);
                if status == 1
                    fprintf('   >> Added fiducial slave to %s\n', image_name);
                    added_count = added_count + 1;
                elseif status == 0
                    fprintf('   >> Replaced existing fiducial slave in %s\n', image_name);
                    added_count = added_count + 1;
                else
                    fprintf('   !! Failed to add fiducial slave (status: %d)\n', status);
                end
            catch ME
                fprintf('   !! Error adding slave to %s: %s\n', image_name, ME.message);
            end
        else
            fprintf('   !! No fiducials to add\n');
        end
    end
    
    % Save the enhanced project
    if added_count > 0
        output_file = '../data/arbuz_integration/enhanced_project_20250804_121653.mat';
        fprintf('\n>> Saving enhanced project: %s\n', output_file);
        
        try
            status = arbuz_SaveProject(hGUI, output_file);
            if status
                fprintf('>> Enhanced project saved successfully\n');
                fprintf('\n** INTEGRATION COMPLETE! **\n');
                fprintf('   Images processed: %d\n', added_count);
                fprintf('   Total fiducials: %d\n', fid_data.total_fiducials);
                fprintf('   Enhanced project: %s\n', output_file);
                fprintf('\n>> You can now open the enhanced project in ArbuzGUI\n');
            else
                fprintf('!! Failed to save enhanced project\n');
            end
        catch ME
            fprintf('!! Error saving project: %s\n', ME.message);
        end
    else
        fprintf('\n!! No fiducials were added - nothing to save\n');
    end
    
    % Clean up
    try
        close(hGUI);
    catch
        % Ignore cleanup errors
    end
    
    fprintf('\n>> Integration script completed\n');
end
