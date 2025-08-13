function compare_slave_structures()
% COMPARE_SLAVE_STRUCTURES - Compare working training file slaves with our created slaves

    fprintf('🔍 Comparing slave structures...\n');
    
    % Check a training file that has working slaves
    training_files = {
        'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\HemoM002.mat',
        'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\HemoM003.mat',
        'C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\ExchangeB6M005.mat'
    };
    
    for f = 1:length(training_files)
        file_path = training_files{f};
        [~, filename, ~] = fileparts(file_path);
        
        if exist(file_path, 'file')
            fprintf('\n📂 Checking: %s\n', filename);
            
            try
                data = load(file_path);
                
                if isfield(data, 'images')
                    images = data.images;
                    
                    for i = 1:min(3, size(images, 2))  % Check first 3 images
                        img = images{1, i};
                        
                        if isfield(img, 'slaves') && ~isempty(img.slaves)
                            fprintf('  Image %d has slaves:\n', i);
                            slaves = img.slaves;
                            
                            if iscell(slaves)
                                for j = 1:min(2, length(slaves))  % Check first 2 slaves
                                    slave = slaves{j};
                                    if isstruct(slave)
                                        slave_name = getfield(slave, 'Name', 'UNKNOWN');
                                        slave_type = getfield(slave, 'ImageType', 'UNKNOWN');
                                        visible = getfield(slave, 'Visible', 'UNKNOWN');
                                        loaded = getfield(slave, 'isLoaded', 'UNKNOWN');
                                        store = getfield(slave, 'isStore', 'UNKNOWN');
                                        
                                        fprintf('    Slave %d: "%s" (%s) - Visible:%s, Loaded:%s, Store:%s\n', ...
                                            j, slave_name, slave_type, mat2str(visible), mat2str(loaded), mat2str(store));
                                        
                                        % Check if it has data
                                        if isfield(slave, 'data')
                                            slave_data = slave.data;
                                            fprintf('      Data: %s, unique values: %s\n', ...
                                                mat2str(size(slave_data)), mat2str(unique(slave_data(:))'));
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                
            catch ME
                fprintf('  Error: %s\n', ME.message);
            end
        end
    end
    
    % Now check our created file
    fprintf('\n📂 Checking our created file:\n');
    our_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\kidney_slaves_20250813_131814\withoutROIwithMRI_FINAL_KIDNEY_SLAVES.mat';
    
    if exist(our_file, 'file')
        try
            data = load(our_file);
            
            if isfield(data, 'images')
                images = data.images;
                img = images{1, 1};  % First image which has the slave
                
                if isfield(img, 'slaves') && ~isempty(img.slaves)
                    slaves = img.slaves;
                    slave = slaves{1};
                    
                    slave_name = getfield(slave, 'Name', 'UNKNOWN');
                    slave_type = getfield(slave, 'ImageType', 'UNKNOWN');
                    visible = getfield(slave, 'Visible', 'UNKNOWN');
                    loaded = getfield(slave, 'isLoaded', 'UNKNOWN');
                    store = getfield(slave, 'isStore', 'UNKNOWN');
                    
                    fprintf('  Our Slave: "%s" (%s) - Visible:%s, Loaded:%s, Store:%s\n', ...
                        slave_name, slave_type, mat2str(visible), mat2str(loaded), mat2str(store));
                    
                    if isfield(slave, 'data')
                        slave_data = slave.data;
                        fprintf('    Data: %s, unique values: %s\n', ...
                            mat2str(size(slave_data)), mat2str(unique(slave_data(:))'));
                    end
                end
            end
            
        catch ME
            fprintf('  Error: %s\n', ME.message);
        end
    end
end

function value = getfield(s, field, default)
    if isfield(s, field)
        value = s.(field);
    else
        value = default;
    end
end

% Run the comparison
compare_slave_structures();
