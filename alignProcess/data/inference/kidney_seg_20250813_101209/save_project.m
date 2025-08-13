% Auto-generated MATLAB script for saving kidney segmentation results
% Generated on 2025-08-13 10:12:09

function save_project()
    try
        fprintf('Loading original data...\n');
        original_data = load('c:/Users/ftmen/Documents/mrialign/alignProcess/data/training/withoutROIwithMRI.mat');
        
        fprintf('Loading processed images data...\n');
        processed_data = load('c:/Users/ftmen/Documents/mrialign/alignProcess/data/inference/kidney_seg_20250813_101209/processed_images.mat');
        
        % Update images in the project
        file_type = original_data.file_type;
        
        % Convert processed images back to proper format
        images = cell(1, length(processed_data.processed_images));
        for i = 1:length(processed_data.processed_images)
            img = processed_data.processed_images(i);
            images{i} = img;
        end
        
        transformations = original_data.transformations;
        sequences = original_data.sequences;
        groups = original_data.groups;
        activesequence = original_data.activesequence;
        activetransformation = original_data.activetransformation;
        saves = original_data.saves;
        
        % Update comments
        if isfield(original_data, 'comments')
            if ischar(original_data.comments)
                comments = [original_data.comments, ' | Mock kidney segmentation added on 2025-08-13 10:12:09'];
            else
                comments = 'Mock kidney segmentation added on 2025-08-13 10:12:09';
            end
        else
            comments = 'Mock kidney segmentation added on 2025-08-13 10:12:09';
        end
        
        status = original_data.status;
        
        % Save the new project file
        fprintf('Saving new project file...\n');
        save('c:/Users/ftmen/Documents/mrialign/alignProcess/data/inference/kidney_seg_20250813_101209/kidney_segmented_withoutROIwithMRI.mat', 'file_type', 'images', 'transformations', ...
             'sequences', 'groups', 'activesequence', 'activetransformation', ...
             'saves', 'comments', 'status', '-v7.3');
        
        fprintf('Successfully saved kidney segmentation results to:\n');
        fprintf('c:/Users/ftmen/Documents/mrialign/alignProcess/data/inference/kidney_seg_20250813_101209/kidney_segmented_withoutROIwithMRI.mat\n');
        
    catch ME
        fprintf('Error saving project: %s\n', ME.message);
        fprintf('Error details: %s\n', ME.getReport);
        exit(1);
    end
end

% Run the function
save_project();
