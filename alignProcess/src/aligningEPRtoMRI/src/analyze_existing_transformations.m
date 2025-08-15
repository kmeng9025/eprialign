function analyze_existing_transformations()
% ANALYZE_EXISTING_TRANSFORMATIONS - Examine the structure of a file with transformations
    
    fprintf('🔍 Analyzing existing transformations structure...\n');
    
    % Load the file with existing transformations
    training_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat';
    
    if ~exist(training_file, 'file')
        error('Training file not found: %s', training_file);
    end
    
    data = load(training_file);
    
    fprintf('📂 Loaded training file: %s\n', training_file);
    
    % Check for type field
    if isfield(data, 'type')
        fprintf('📊 File type: %s\n', data.type);
    else
        fprintf('📊 File type: Not specified\n');
    end
    
    % Analyze main structure
    fprintf('\n📋 Main structure fields:\n');
    fields = fieldnames(data);
    for i = 1:length(fields)
        field = fields{i};
        field_data = data.(field);
        
        if isstruct(field_data)
            fprintf('  %s: struct with %d field(s)\n', field, length(fieldnames(field_data)));
        elseif iscell(field_data)
            fprintf('  %s: cell array %s\n', field, mat2str(size(field_data)));
        else
            fprintf('  %s: %s %s\n', field, class(field_data), mat2str(size(field_data)));
        end
    end
    
    % Analyze transformations if they exist
    if isfield(data, 'transformations') && ~isempty(data.transformations)
        fprintf('\n🔄 TRANSFORMATIONS ANALYSIS:\n');
        transformations = data.transformations;
        fprintf('  Number of transformations: %d\n', length(transformations));
        
        for i = 1:length(transformations)
            trans = transformations{i};
            fprintf('\n  Transformation %d:\n', i);
            
            if isfield(trans, 'Name')
                fprintf('    Name: %s\n', trans.Name);
            end
            
            if isfield(trans, 'Anative')
                fprintf('    Anative matrix: %s\n', mat2str(size(trans.Anative)));
                fprintf('    Anative:\n');
                disp(trans.Anative);
            end
            
            if isfield(trans, 'Comments')
                fprintf('    Comments: %s\n', trans.Comments);
            end
            
            % Show all fields
            trans_fields = fieldnames(trans);
            fprintf('    All fields: %s\n', strjoin(trans_fields, ', '));
        end
    else
        fprintf('\n⚠️  No transformations found in this file\n');
    end
    
    % Analyze sequences if they exist
    if isfield(data, 'sequences') && ~isempty(data.sequences)
        fprintf('\n📚 SEQUENCES ANALYSIS:\n');
        sequences = data.sequences;
        fprintf('  Number of sequences: %d\n', length(sequences));
        
        for i = 1:length(sequences)
            seq = sequences{i};
            fprintf('\n  Sequence %d:\n', i);
            
            if isfield(seq, 'Name')
                fprintf('    Name: %s\n', seq.Name);
            end
            
            % Show all fields
            seq_fields = fieldnames(seq);
            fprintf('    All fields: %s\n', strjoin(seq_fields, ', '));
        end
    else
        fprintf('\n⚠️  No sequences found in this file\n');
    end
    
    % Analyze images structure
    if isfield(data, 'images') && ~isempty(data.images)
        fprintf('\n🖼️  IMAGES ANALYSIS:\n');
        images = data.images;
        fprintf('  Number of images: %d\n', size(images, 2));
        
        for i = 1:min(3, size(images, 2))  % Show first 3 images
            img = images{1, i};
            fprintf('\n  Image %d:\n', i);
            
            if isfield(img, 'Name')
                fprintf('    Name: %s\n', img.Name);
            end
            
            if isfield(img, 'data') && ~isempty(img.data)
                fprintf('    Data size: %s\n', mat2str(size(img.data)));
            end
            
            if isfield(img, 'A')
                fprintf('    A matrix: %s\n', mat2str(size(img.A)));
            end
            
            if isfield(img, 'Anative')
                fprintf('    Anative matrix: %s\n', mat2str(size(img.Anative)));
            end
            
            % Show all fields
            img_fields = fieldnames(img);
            fprintf('    All fields: %s\n', strjoin(img_fields, ', '));
        end
    end
    
    fprintf('\n✅ Analysis complete!\n');
end
