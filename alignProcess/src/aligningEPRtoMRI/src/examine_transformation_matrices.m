% Examine transformation matrices in detail
fprintf('Examining transformation matrices in detail...\n');

% Load the training file
training_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat';
data = load(training_file);

fprintf('Loaded training file: %s\n', training_file);

% Look at image transformation matrices
fprintf('\nIMAGE TRANSFORMATION MATRICES:\n');
for i = 1:length(data.images)
    img = data.images{i};
    fprintf('\nImage %d: %s\n', i, img.Name);
    
    % Native transformation matrix
    fprintf('  Anative (native coordinates):\n');
    disp(img.Anative);
    
    % Current transformation matrix
    fprintf('  A (current transformation):\n');
    disp(img.A);
    
    % Prime transformation matrix
    if isfield(img, 'Aprime') && ~isempty(img.Aprime)
        fprintf('  Aprime (prime transformation):\n');
        disp(img.Aprime);
    end
    
    % Image data info
    if isfield(img, 'data_info')
        fprintf('  Data info:\n');
        disp(img.data_info);
    end
end

% Look at transformation objects
fprintf('\nTRANSFORMATION OBJECTS:\n');
for i = 1:length(data.transformations)
    trans = data.transformations{i};
    fprintf('\nTransformation %d: %s\n', i, trans.Name);
    
    fprintf('  Matrices field:\n');
    if isfield(trans, 'Matrices') && ~isempty(trans.Matrices)
        % Show the matrices structure
        matrices = trans.Matrices;
        fprintf('    Number of matrices: %d\n', length(matrices));
        
        % Show first few matrices
        for j = 1:min(3, length(matrices))
            fprintf('    Matrix %d:\n', j);
            disp(matrices{j});
        end
    end
end

fprintf('\nExamination complete!\n');
