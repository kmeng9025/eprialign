function combine_arbuz_with_ai(original_file, ai_results_file, output_file)
% COMBINE_ARBUZ_WITH_AI - Create Arbuz-compatible file with AI kidney results
%
% Usage:
%   combine_arbuz_with_ai('original.mat', 'ai_results.mat', 'output.mat')
%
% This function properly combines an original Arbuz project file with AI results
% maintaining full Arbuz compatibility for loading in ArbuzGUI

    fprintf('🔄 Combining Arbuz project with AI kidney results...\n');
    fprintf('   📂 Original Arbuz file: %s\n', original_file);
    fprintf('   🤖 AI results file: %s\n', ai_results_file);
    fprintf('   💾 Output file: %s\n', output_file);
    
    try
        % Load original Arbuz project (preserves all structure)
        fprintf('   📂 Loading original Arbuz project...\n');
        original = load(original_file);
        
        % Load AI results
        fprintf('   🤖 Loading AI results...\n');
        ai_results = load(ai_results_file);
        
        % Start with complete original Arbuz structure
        output_data = original;
        
        % Add AI results to the structure
        fprintf('   🔗 Adding AI results to Arbuz structure...\n');
        
        % Get all AI fields and add them
        ai_fields = fieldnames(ai_results);
        for i = 1:length(ai_fields)
            field_name = ai_fields{i};
            output_data.(field_name) = ai_results.(field_name);
            fprintf('     ✅ Added: %s\n', field_name);
        end
        
        % Save combined file using MATLAB's native format (preserves Arbuz compatibility)
        fprintf('   💾 Saving Arbuz-compatible file...\n');
        save(output_file, '-struct', 'output_data', '-v7.3');
        
        % Display summary
        fprintf('✅ Arbuz-compatible file created successfully!\n');
        fprintf('   📁 File: %s\n', output_file);
        
        % Show file size
        file_info = dir(output_file);
        fprintf('   📊 File size: %.1f MB\n', file_info.bytes / (1024*1024));
        
        % Display AI summary if available
        if isfield(output_data, 'ai_num_kidneys_detected')
            fprintf('   👁️  Kidneys detected: %d\n', output_data.ai_num_kidneys_detected);
        end
        if isfield(output_data, 'ai_coverage_percent')
            fprintf('   📊 Kidney coverage: %.2f%%\n', output_data.ai_coverage_percent);
        end
        if isfield(output_data, 'ai_training_f1_score')
            fprintf('   🎯 Model F1 score: %.3f\n', output_data.ai_training_f1_score);
        end
        
        fprintf('📋 File is now ready for ArbuzGUI!\n');
        
    catch ME
        fprintf('❌ Error combining files: %s\n', ME.message);
        fprintf('   Error details: %s\n', ME.getReport);
        rethrow(ME);
    end
end
