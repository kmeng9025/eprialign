% Examine the box data structure
fprintf('Examining box data structure...\n');

% Load the fiducials file
fiducials_file = 'C:\Users\ftmen\Documents\mrialign\alignProcess\data\inference\fiducials_withoutROIwithMRIwithoutTransformations_20250814_125342\box_fiducials_withoutROIwithMRIwithoutTransformations_20250814_125345\withoutROIwithMRIwithoutTransformations_with_boxes.mat';
data = load(fiducials_file);

fprintf('Examining boxes in each image...\n');

for i = 1:length(data.images)
    img = data.images{i};
    fprintf('\nImage %d: %s\n', i, img.Name);
    
    if isfield(img, 'slaves') && ~isempty(img.slaves)
        slave = img.slaves{1}; % Should only be one slave per image
        
        fprintf('  Slave name: %s\n', slave.Name);
        
        if isfield(slave, 'box')
            box = slave.box;
            fprintf('  Box class: %s\n', class(box));
            fprintf('  Box size: [%s]\n', num2str(size(box)));
            
            if isnumeric(box)
                fprintf('  Box contents:\n');
                disp(box);
            elseif iscell(box)
                fprintf('  Box is cell array with %d elements\n', length(box));
                
                for j = 1:min(5, length(box))
                    fprintf('    Box %d: %s, size [%s]\n', j, class(box{j}), num2str(size(box{j})));
                    if isnumeric(box{j})
                        fprintf('      Content: %s\n', mat2str(box{j}));
                    end
                end
            end
        else
            fprintf('  No box field found\n');
        end
        
        % Also check if there's a 'data' field
        if isfield(slave, 'data')
            fprintf('  Data field: %s, size [%s]\n', class(slave.data), num2str(size(slave.data)));
        end
    end
end

fprintf('\nExamination complete!\n');
