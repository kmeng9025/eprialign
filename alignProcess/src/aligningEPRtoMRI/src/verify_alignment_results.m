function verify_alignment_results()
    % Verify the alignment results by examining the transformation matrices
    
    fprintf('Verifying alignment results...\n');
    
    % Load the original and aligned data
    original_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat';
    aligned_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI_aligned.mat';
    
    original_data = load(original_file);
    aligned_data = load(aligned_file);
    
    fprintf('Loaded original and aligned data files\n');
    
    % Compare transformation matrices
    epr_images = {'>PreHemo', '>Hemo', '>Post_Transfus2'};
    
    for i = 1:length(epr_images)
        img_name = epr_images{i};
        
        % Find the image in both datasets
        orig_img = [];
        aligned_img = [];
        
        for j = 1:length(original_data.images)
            if strcmp(original_data.images{j}.Name, img_name)
                orig_img = original_data.images{j};
                break;
            end
        end
        
        for j = 1:length(aligned_data.images)
            if strcmp(aligned_data.images{j}.Name, img_name)
                aligned_img = aligned_data.images{j};
                break;
            end
        end
        
        if ~isempty(orig_img) && ~isempty(aligned_img)
            fprintf('\n%s Transformation Comparison:\n', img_name);
            
            fprintf('  Original A matrix:\n');
            disp(orig_img.A);
            
            fprintf('  Aligned A matrix:\n');
            disp(aligned_img.A);
            
            % Calculate the applied transformation
            transformation = aligned_img.A * inv(orig_img.A);
            fprintf('  Applied transformation (A_new * A_orig^-1):\n');
            disp(transformation);
            
            % Extract scale, rotation, and translation
            scale_x = norm(transformation(1:3, 1));
            scale_y = norm(transformation(1:3, 2));
            scale_z = norm(transformation(1:3, 3));
            translation = transformation(1:3, 4);
            
            fprintf('  Scale factors: [%.4f, %.4f, %.4f]\n', scale_x, scale_y, scale_z);
            fprintf('  Translation: [%.4f, %.4f, %.4f]\n', translation(1), translation(2), translation(3));
        else
            fprintf('  Could not find %s in both datasets\n', img_name);
        end
    end
    
    fprintf('\nVerification complete!\n');
end
