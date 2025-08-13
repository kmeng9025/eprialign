function run_ai_kidney_detection_pipeline(input_file, varargin)
% RUN_AI_KIDNEY_DETECTION_PIPELINE - Complete AI kidney detection with Arbuz output
%
% Usage:
%   run_ai_kidney_detection_pipeline('input.mat')
%   run_ai_kidney_detection_pipeline('input.mat', 'OutputDir', 'custom_dir')
%
% This function runs the complete AI kidney detection pipeline and creates
% Arbuz-compatible output files ready for loading in ArbuzGUI.

    % Parse inputs
    p = inputParser;
    addRequired(p, 'input_file', @ischar);
    addParameter(p, 'OutputDir', '', @ischar);
    parse(p, input_file, varargin{:});
    
    input_file = p.Results.input_file;
    output_dir = p.Results.OutputDir;
    
    fprintf('🚀 AI KIDNEY DETECTION PIPELINE (MATLAB)\n');
    fprintf('==================================================\n');
    fprintf('📁 Input file: %s\n', input_file);
    
    % Check if input file exists
    if ~exist(input_file, 'file')
        error('Input file does not exist: %s', input_file);
    end
    
    % Set default output directory
    if isempty(output_dir)
        script_dir = fileparts(mfilename('fullpath'));
        output_dir = fullfile(script_dir, '..', '..', 'data', 'inference');
    end
    
    fprintf('📁 Output directory: %s\n', output_dir);
    
    try
        % Step 1: Run Python AI pipeline
        fprintf('\n🐍 STEP 1: Running Python AI Detection...\n');
        fprintf('=========================================\n');
        
        script_dir = fileparts(mfilename('fullpath'));
        python_script = fullfile(script_dir, 'ai_kidney_pipeline_with_mat_output.py');
        
        % Prepare Python command
        cmd = sprintf('python "%s" "%s" "%s"', python_script, input_file, output_dir);
        fprintf('   Command: %s\n', cmd);
        
        % Execute Python script
        [status, result] = system(cmd);
        
        if status ~= 0
            error('Python AI pipeline failed:\n%s', result);
        end
        
        fprintf('✅ Python AI detection completed successfully!\n');
        fprintf('Python output:\n%s\n', result);
        
        % Step 2: Find the AI results file
        fprintf('\n🔍 STEP 2: Locating AI Results...\n');
        fprintf('==================================\n');
        
        % Parse the Python output to find the output directory
        lines = strsplit(result, '\n');
        ai_results_file = '';
        
        for i = 1:length(lines)
            line = lines{i};
            if contains(line, '_with_AI_kidneys.mat') && contains(line, 'File:')
                % Extract filename from the line
                parts = strsplit(line, 'File: ');
                if length(parts) > 1
                    ai_results_file = strtrim(parts{2});
                    break;
                end
            end
        end
        
        if isempty(ai_results_file)
            error('Could not find AI results file in Python output');
        end
        
        fprintf('📄 Found AI results: %s\n', ai_results_file);
        
        % Step 3: Create Arbuz-compatible file
        fprintf('\n🔗 STEP 3: Creating Arbuz-Compatible File...\n');
        fprintf('=============================================\n');
        
        [ai_dir, ai_name, ~] = fileparts(ai_results_file);
        arbuz_output = fullfile(ai_dir, strrep(ai_name, '_with_AI_kidneys', '_ARBUZ_COMPATIBLE.mat'));
        
        fprintf('   Combining:\n');
        fprintf('   📂 Original: %s\n', input_file);
        fprintf('   🤖 AI results: %s\n', ai_results_file);
        fprintf('   💾 Arbuz output: %s\n', arbuz_output);
        
        % Use our combination function
        combine_arbuz_with_ai(input_file, ai_results_file, arbuz_output);
        
        % Step 4: Summary
        fprintf('\n🎉 PIPELINE COMPLETE!\n');
        fprintf('====================\n');
        fprintf('✅ AI kidney detection completed successfully!\n');
        fprintf('📁 Output directory: %s\n', ai_dir);
        fprintf('📄 Files created:\n');
        
        % List all files in output directory
        output_files = dir(ai_dir);
        for i = 1:length(output_files)
            if ~output_files(i).isdir
                fprintf('   📄 %s (%.1f MB)\n', output_files(i).name, output_files(i).bytes/(1024*1024));
            end
        end
        
        fprintf('\n🎯 READY FOR ARBUZGUI:\n');
        fprintf('   📂 Open this file: %s\n', arbuz_output);
        fprintf('   🏥 Contains AI kidney detection with original Arbuz structure\n');
        
        % Return the Arbuz-compatible file path
        if nargout > 0
            varargout{1} = arbuz_output;
        end
        
    catch ME
        fprintf('❌ Pipeline failed: %s\n', ME.message);
        fprintf('   Error details: %s\n', ME.getReport);
        rethrow(ME);
    end
end
