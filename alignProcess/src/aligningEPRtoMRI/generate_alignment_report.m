function generate_alignment_report(transformations, report_file)
    % Generate a detailed alignment report
    
    fprintf('Generating alignment report: %s\n', report_file);
    
    fid = fopen(report_file, 'w');
    if fid == -1
        error('Could not create report file: %s', report_file);
    end
    
    try
        fprintf(fid, 'EPR-to-MRI Alignment Report\n');
        fprintf(fid, '===========================\n\n');
        fprintf(fid, 'Generated on: %s\n\n', datestr(now));
        
        fprintf(fid, 'Summary:\n');
        fprintf(fid, '--------\n');
        fprintf(fid, 'Number of aligned images: %d\n\n', length(transformations));
        
        if isempty(transformations)
            fprintf(fid, 'No transformations were calculated.\n');
            fprintf(fid, 'This may be due to insufficient fiducials (need at least 3 per image).\n\n');
        else
            fprintf(fid, 'Transformation Details:\n');
            fprintf(fid, '----------------------\n\n');
            
            for i = 1:length(transformations)
                trans = transformations{i};
                T = trans.matrix;
                
                fprintf(fid, '%s:\n', trans.name);
                
                % Extract transformation components
                scale_x = norm(T(1:3, 1));
                scale_y = norm(T(1:3, 2));
                scale_z = norm(T(1:3, 3));
                translation = T(1:3, 4);
                
                fprintf(fid, '  Scale factors:\n');
                fprintf(fid, '    X: %.6f\n', scale_x);
                fprintf(fid, '    Y: %.6f\n', scale_y);
                fprintf(fid, '    Z: %.6f\n', scale_z);
                fprintf(fid, '    Average: %.6f\n', mean([scale_x, scale_y, scale_z]));
                
                fprintf(fid, '  Translation (voxels):\n');
                fprintf(fid, '    X: %.4f\n', translation(1));
                fprintf(fid, '    Y: %.4f\n', translation(2));
                fprintf(fid, '    Z: %.4f\n', translation(3));
                
                % Calculate rotation angles (approximate)
                R = T(1:3, 1:3) / scale_x; % Remove scaling to get rotation
                
                % Extract Euler angles (ZYX convention)
                sy = sqrt(R(1,1)^2 + R(2,1)^2);
                singular = sy < 1e-6;
                
                if ~singular
                    x_angle = atan2(R(3,2), R(3,3));
                    y_angle = atan2(-R(3,1), sy);
                    z_angle = atan2(R(2,1), R(1,1));
                else
                    x_angle = atan2(-R(2,3), R(2,2));
                    y_angle = atan2(-R(3,1), sy);
                    z_angle = 0;
                end
                
                fprintf(fid, '  Rotation (degrees):\n');
                fprintf(fid, '    X: %.2f\n', rad2deg(x_angle));
                fprintf(fid, '    Y: %.2f\n', rad2deg(y_angle));
                fprintf(fid, '    Z: %.2f\n', rad2deg(z_angle));
                
                fprintf(fid, '  Transformation Matrix:\n');
                for row = 1:4
                    fprintf(fid, '    [');
                    for col = 1:4
                        fprintf(fid, '%10.6f', T(row, col));
                        if col < 4
                            fprintf(fid, ', ');
                        end
                    end
                    fprintf(fid, ']\n');
                end
                
                fprintf(fid, '\n');
            end
        end
        
        fprintf(fid, 'Notes:\n');
        fprintf(fid, '------\n');
        fprintf(fid, '- Scale factors represent the size ratio between EPR and MRI voxels\n');
        fprintf(fid, '- Translation values are in the target (MRI) coordinate system\n');
        fprintf(fid, '- Rotation angles are approximate Euler angles in degrees\n');
        fprintf(fid, '- The transformation matrix is applied as: MRI_coords = T * EPR_coords\n');
        
    catch ME
        fclose(fid);
        rethrow(ME);
    end
    
    fclose(fid);
    fprintf('Report saved successfully.\n');
end
