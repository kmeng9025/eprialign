function run_ai_kidney_pipeline_matlab(input_mat_file, output_dir)
% RUN_AI_KIDNEY_PIPELINE_MATLAB - Complete AI kidney segmentation pipeline in MATLAB
%
% Inputs:
%   input_mat_file - Path to input .mat file with MRI data
%   output_dir - Output directory for results (optional, defaults to inference folder)
%
% This function calls the Python AI pipeline and saves Arbuz-compatible results

    if nargin < 2
        % Default output directory
        output_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data', 'inference');
    end
    
    fprintf('🔬 Starting AI Kidney Segmentation Pipeline (MATLAB Version)\n');
    fprintf('📁 Input file: %s\n', input_mat_file);
    fprintf('📁 Output directory: %s\n', output_dir);
    
    % Check if input file exists
    if ~exist(input_mat_file, 'file')
        error('Input file does not exist: %s', input_mat_file);
    end
    
    % Create output directory if it doesn't exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Create timestamped subdirectory
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    output_subdir = fullfile(output_dir, ['kidney_detection_' timestamp]);
    mkdir(output_subdir);
    
    fprintf('📂 Created output directory: %s\n', output_subdir);
    
    try
        % Call Python AI pipeline
        python_script = fullfile(fileparts(mfilename('fullpath')), 'ai_kidney_pipeline_with_mat_output.py');
        
        % Prepare command
        cmd = sprintf('python "%s" "%s" "%s"', python_script, input_mat_file, output_subdir);
        
        fprintf('🐍 Calling Python AI pipeline...\n');
        fprintf('   Command: %s\n', cmd);
        
        % Execute Python script
        [status, result] = system(cmd);
        
        if status == 0
            fprintf('✅ AI pipeline completed successfully!\n');
            fprintf('📋 Python output:\n%s\n', result);
            
            % List output files
            fprintf('📁 Output files created:\n');
            output_files = dir(fullfile(output_subdir, '*'));
            for i = 1:length(output_files)
                if ~output_files(i).isdir
                    fprintf('   📄 %s (%.1f KB)\n', output_files(i).name, output_files(i).bytes/1024);
                end
            end
            
            % Find the AI results .mat file
            mat_files = dir(fullfile(output_subdir, '*_with_AI_kidneys.mat'));
            if ~isempty(mat_files)
                result_file = fullfile(output_subdir, mat_files(1).name);
                fprintf('🎯 AI results saved to: %s\n', result_file);
                
                % Load and display summary
                try
                    ai_results = load(result_file);
                    if isfield(ai_results, 'ai_num_kidneys_detected')
                        fprintf('👁️  Kidneys detected: %d\n', ai_results.ai_num_kidneys_detected);
                    end
                    if isfield(ai_results, 'ai_coverage_percent')
                        fprintf('📊 Kidney coverage: %.2f%%\n', ai_results.ai_coverage_percent);
                    end
                    if isfield(ai_results, 'ai_training_f1_score')
                        fprintf('🎯 Model F1 score: %.3f\n', ai_results.ai_training_f1_score);
                    end
                catch
                    fprintf('⚠️  Could not load AI results for summary\n');
                end
            end
            
        else
            error('Python AI pipeline failed with status %d:\n%s', status, result);
        end
        
    catch ME
        fprintf('❌ Error in AI pipeline: %s\n', ME.message);
        rethrow(ME);
    end
    
    fprintf('🏁 AI kidney segmentation pipeline completed!\n');
    fprintf('📂 Results available in: %s\n', output_subdir);
end
