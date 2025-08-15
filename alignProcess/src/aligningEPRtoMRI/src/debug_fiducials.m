% Debug fiducials extraction
fprintf('Debugging fiducials extraction...\n');

% Load the fiducials file
fiducials_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\fiducials_withoutROIwithMRIwithoutTransformations_20250814_125342\box_fiducials_withoutROIwithMRIwithoutTransformations_20250814_125345\withoutROIwithMRIwithoutTransformations_with_boxes.mat';
data = load(fiducials_file);

fprintf('Loaded file: %s\n', fiducials_file);

% Show top-level structure
fprintf('\nTop-level fields:\n');
fields = fieldnames(data);
for i = 1:length(fields)
    field = fields{i};
    value = data.(field);
    fprintf('  %s: %s\n', field, class(value));
    if isstruct(value)
        fprintf('    Size: [%s]\n', num2str(size(value)));
    elseif iscell(value)
        if ~isempty(value)
            fprintf('    Size: [%s], Contents: %s\n', num2str(size(value)), class(value{1}));
        else
            fprintf('    Size: [%s], Contents: empty\n', num2str(size(value)));
        end
    else
        fprintf('    Size: [%s]\n', num2str(size(value)));
    end
end

% Look at images
if isfield(data, 'images')
    fprintf('\nImage analysis:\n');
    for i = 1:length(data.images)
        img = data.images{i};
        fprintf('  Image %d: %s\n', i, img.Name);
        
        % Check for slaves
        if isfield(img, 'slaves')
            fprintf('    Slaves: %s, Length: %d\n', class(img.slaves), length(img.slaves));
            
            if ~isempty(img.slaves)
                for j = 1:min(3, length(img.slaves))
                    slave = img.slaves{j};
                    fprintf('      Slave %d fields: ', j);
                    if isstruct(slave)
                        slave_fields = fieldnames(slave);
                        fprintf('%s\n', strjoin(slave_fields, ', '));
                        
                        % Check for center field
                        if isfield(slave, 'center')
                            fprintf('        Center: [%s]\n', num2str(slave.center));
                        end
                    else
                        fprintf('%s\n', class(slave));
                    end
                end
            end
        else
            fprintf('    No slaves field\n');
        end
    end
end

fprintf('\nDebugging complete!\n');
