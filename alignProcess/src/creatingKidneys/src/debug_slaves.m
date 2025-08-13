function debug_slave_creation()
% DEBUG_SLAVE_CREATION - Debug the slave creation process

    % Test with the latest output file
    output_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\kidney_slaves_20250813_131814\withoutROIwithMRI_FINAL_KIDNEY_SLAVES.mat';
    
    fprintf('🔍 Debugging slave creation in: %s\n', output_file);
    
    if exist(output_file, 'file')
        try
            data = load(output_file);
            
            fprintf('📋 Main fields: %s\n', strjoin(fieldnames(data), ', '));
            
            if isfield(data, 'images')
                images = data.images;
                fprintf('🖼️  Images structure: %s\n', mat2str(size(images)));
                
                for i = 1:size(images, 2)
                    img = images{1, i};
                    fprintf('\n  Image %d:\n', i);
                    
                    if isstruct(img)
                        img_fields = fieldnames(img);
                        fprintf('    Fields: %s\n', strjoin(img_fields, ', '));
                        
                        if isfield(img, 'slaves')
                            slaves = img.slaves;
                            if isempty(slaves)
                                fprintf('    ❌ slaves field is EMPTY\n');
                            else
                                fprintf('    ✅ slaves field exists: %s\n', mat2str(size(slaves)));
                                
                                if iscell(slaves)
                                    fprintf('    Slaves is cell array with %d elements\n', length(slaves));
                                    for j = 1:length(slaves)
                                        slave = slaves{j};
                                        if isstruct(slave)
                                            slave_name = getfield(slave, 'Name', 'UNKNOWN');
                                            slave_type = getfield(slave, 'ImageType', 'UNKNOWN');
                                            fprintf('      Slave %d: "%s" (%s)\n', j, slave_name, slave_type);
                                        end
                                    end
                                else
                                    fprintf('    ⚠️  slaves is not a cell array: %s\n', class(slaves));
                                end
                            end
                        else
                            fprintf('    ❌ No slaves field found\n');
                        end
                        
                        if isfield(img, 'data')
                            fprintf('    data: %s\n', mat2str(size(img.data)));
                        end
                    end
                end
            end
            
        catch ME
            fprintf('❌ Error: %s\n', ME.message);
        end
    else
        fprintf('❌ File not found: %s\n', output_file);
    end
end

function value = getfield(s, field, default)
    if isfield(s, field)
        value = s.(field);
    else
        value = default;
    end
end

% Run the debug
debug_slave_creation();
