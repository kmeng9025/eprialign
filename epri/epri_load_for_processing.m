% EPRI_LOAD_FOR_PROCESSING - load single or multiple image data
% [mat,mat_bl,out_parameters] = EPRI_LOAD_FOR_PROCESSING(file_name, load_opt);
% file_name    - Source data filename [string or cell array of strings] 
% load_opt - Loading options [structure]
%   [].to be done   - FBP structure parameters []
% mat               - Data [array nPoints x nTypes x nPrj]
% mat_bl            - Baseline for data [array nPoints x nTypes x nPrj]
% out_parameters    - Return parameter structure
%   [].Raw          - Loaded image [array, 3D or 4D]
%   [].raw_info     - Parameters of loaded projections [structure] 
% See also EPRI_READIMAGEFILE, EPRI_ESE_PREPROCESS, EPRI_RECONSTRUCT.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2014
% Contact: epri.uchicago.edu

function [out,out_parameters] = epri_load_for_processing(file_name, load_opt)

% get file extenstion
if iscell(file_name), [~, ~, fext]=fileparts(file_name{1});
else [~, ~, fext]=fileparts(file_name);
end

out_parameters = [];

try
  n = 0;
  switch lower(fext)
    case {'.d01', '.tdms', '.dsc', '.dta'}
      if iscell(file_name)
        % Multiple files         
        for ii=1:length(file_name)
          [out,out_parameters.raw_info]=epri_ReadImageFile(file_name{ii}, load_opt);
          if ii==1, mat = out.mat; mat_bl = out.mat_bl;
          else, mat = mat + out.mat; mat_bl = mat_bl + out.mat_bl;
          end
          n = n + 1;
        end
        out.mat = mat / length(file_name); out.mat_bl = mat_bl / length(file_name);
      else
        % Single file
        [out,out_parameters.raw_info] = epri_ReadImageFile(file_name, load_opt);
        n = 1;
      end
    case '.mat'
      try
        if iscell(file_name)
          file_type = load(file_name{1}, 'file_type');
          loaded_image = load(file_name{1});
        else
          file_type = load(file_name, 'file_type');
          loaded_image = load(file_name);
        end
        file_type = file_type.file_type;
      catch err
        file_type = '?';
      end
      
      out.mat = [];
      out.mat_bl = [];
      switch file_type
        case 'Projections_v1.0'
          n = 1;
        otherwise
          out_parameters.Raw = loaded_image.mat_recFXD;
          out_parameters.raw_info = safeget(loaded_image, 'raw_info', []);
          out_parameters.rec_info = safeget(loaded_image, 'rec_info', []);
          out_parameters.Size = out_parameters.rec_info.rec.size;
          if isfield(out_parameters, 'mat')
            out.mat = out_parameters.mat;
          end
          n = 1;
      end
    case '.img'
      [out.mat, out_parameters.raw_info]=epr_ReadCWImageFile(file_name, load_opt);
      n = 1;
    case ''
      error('File type is not supported');
  end
  disp(sprintf('%i data file(s) loaded.', n));
catch err
  out.mat = []; 
  out.mat_bl = []; 
  disp(err.message);
end