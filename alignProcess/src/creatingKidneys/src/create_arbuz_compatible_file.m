function create_arbuz_compatible_file(original_file, ai_results_file, output_file)
% CREATE_ARBUZ_COMPATIBLE_FILE - Combine original Arbuz project with AI results
%
% Inputs:
%   original_file - Path to original Arbuz .mat file
%   ai_results_file - Path to AI results .mat file (Python output)
%   output_file - Output path for combined file
%
% This MATLAB script properly preserves Arbuz project structure

    fprintf('🔄 Creating Arbuz-compatible file...\n');
    fprintf('   📂 Original: %s\n', original_file);
    fprintf('   🤖 AI results: %s\n', ai_results_file);
    fprintf('   💾 Output: %s\n', output_file);
    
    try
        % Load original Arbuz project structure
        fprintf('   📂 Loading original Arbuz structure...\n');
        original_data = load(original_file);
        
        % Load AI results
        fprintf('   🤖 Loading AI results...\n');
        ai_data = load(ai_results_file);
        
        % Start with complete original structure
        output_data = original_data;
        
        % Add AI results while preserving Arbuz compatibility
        fprintf('   🔗 Merging AI results with Arbuz structure...\n');
        
        % Add AI fields from the AI results file
        ai_fields = fieldnames(ai_data);
        for i = 1:length(ai_fields)
            field_name = ai_fields{i};
            if startsWith(field_name, 'ai_')
                output_data.(field_name) = ai_data.(field_name);
                fprintf('     ✅ Added field: %s\n', field_name);
            end
        end
        
        % Save combined file using MATLAB's native format
        fprintf('   💾 Saving Arbuz-compatible file...\n');
        save(output_file, '-struct', 'output_data', '-v7.3');
        
        fprintf('✅ Arbuz-compatible file created successfully!\n');
        fprintf('   📁 File: %s\n', output_file);
        
        % Display summary
        if isfield(output_data, 'ai_num_kidneys_detected')
            fprintf('   👁️  Kidneys detected: %d\n', output_data.ai_num_kidneys_detected);
        end
        if isfield(output_data, 'ai_coverage_percent')
            fprintf('   📊 Coverage: %.2f%%\n', output_data.ai_coverage_percent);
        end
        
    catch ME
        fprintf('❌ Error creating Arbuz-compatible file: %s\n', ME.message);
        rethrow(ME);
    end
end
