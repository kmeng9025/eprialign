% INSPECT_TRAINING_SLAVES - Examine slave structure in training files
%
% This script inspects the slave structure in training .mat files
% to understand how to properly create kidney slaves
    fprintf('🔍 Inspecting slave structure in training files...\n');
    
    % Training files to inspect
    training_files = {
        'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat',
        'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withROIwithMRI.mat'
    };
    
    for i = 1:length(training_files)
        file_path = training_files{i};
        [~, filename, ~] = fileparts(file_path);
        
        fprintf('\n📂 Analyzing: %s\n', filename);
        fprintf('=' + repmat('=', 1, 50) + '\n');
        
        if exist(file_path, 'file')
            try
                data = load(file_path);
                
                % Display main structure
                fields = fieldnames(data);
                fprintf('📋 Main fields: %s\n', strjoin(fields, ', '));
                
                % Look for slaves
                if isfield(data, 'slaves')
                    slaves = data.slaves;
                    fprintf('🎯 SLAVES FOUND!\n');
                    fprintf('   Structure: %s\n', mat2str(size(slaves)));
                    
                    % Examine slave structure
                    if ~isempty(slaves)
                        fprintf('   Number of slaves: %d\n', numel(slaves));
                        
                        % Look at first slave
                        if numel(slaves) >= 1
                            slave1 = slaves(1);
                            slave_fields = fieldnames(slave1);
                            fprintf('   First slave fields: %s\n', strjoin(slave_fields, ', '));
                            
                            % Check for important fields
                            for field = {'name', 'type', 'data', 'mask', 'points', 'color'}
                                if isfield(slave1, field{1})
                                    value = slave1.(field{1});
                                    if ischar(value) || isstring(value)
                                        fprintf('     %s: "%s"\n', field{1}, value);
                                    elseif isnumeric(value)
                                        fprintf('     %s: %s\n', field{1}, mat2str(size(value)));
                                    else
                                        fprintf('     %s: %s\n', field{1}, class(value));
                                    end
                                end
                            end
                        end
                        
                        % Look at all slaves
                        for j = 1:min(numel(slaves), 5) % Show up to 5 slaves
                            slave = slaves(j);
                            if isfield(slave, 'name')
                                fprintf('   Slave %d: "%s"\n', j, slave.name);
                            else
                                fprintf('   Slave %d: (no name)\n', j);
                            end
                        end
                    else
                        fprintf('   ⚠️  Slaves field exists but is empty\n');
                    end
                else
                    fprintf('   ❌ No slaves field found\n');
                end
                
                % Look for images to understand structure
                if isfield(data, 'images')
                    images = data.images;
                    fprintf('🖼️  Images structure: %s\n', mat2str(size(images)));
                    
                    if ~isempty(images)
                        img1 = images(1);
                        img_fields = fieldnames(img1);
                        fprintf('   Image fields: %s\n', strjoin(img_fields, ', '));
                    end
                end
                
                % Look for sequences
                if isfield(data, 'sequences')
                    sequences = data.sequences;
                    fprintf('📺 Sequences structure: %s\n', mat2str(size(sequences)));
                end
                
            catch ME
                fprintf('❌ Error loading %s: %s\n', filename, ME.message);
            end
        else
            fprintf('❌ File not found: %s\n', file_path);
        end
    end

fprintf('\n✅ Slave structure inspection complete!\n');
