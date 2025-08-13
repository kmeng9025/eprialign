% EPRI_FILENAME  filenames for reconstruction routines
% fnames = EPRI_FILENAME(file_name);
% file_name  - Source data filename [string or array of strings] 
% file_suffix  - Any addition to file name [string] 
% output_path  - Output path (can be empty) [string] 
% fnames - Output filenames [structure]
%   [].path         - File folder [string]
%   [].raw_file     - Raw file name [string]
%   [].p_file       - Fitted data file name [string]
%   [].prj_file     - Projection file name [string] 
% See also EPRI_LOAD_FOR_PROCESSING.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2014
% Contact: epri.uchicago.edu

function fnames = epri_filename(file_name, file_suffix, output_path)

if iscell(file_name), [fp, fn, fext]=fileparts(file_name{1});
else [fp, fn, fext]=fileparts(file_name);
end

if strfind(fn, 'prj_') == 1, 
  fn = fn(5:end);
end

if isempty(output_path)
  fnames.path = fp;
  fnames.raw_file = fullfile(fp, [fn, file_suffix,'.mat']);
  fnames.p_file = fullfile(fp, ['p',fn, file_suffix,'.mat']);
  fnames.prj_file = fullfile(fp, ['prj_',fn, file_suffix,'.mat']);
  fnames.prj_rawfile = fullfile(fp, ['raw_',fn, file_suffix,'.mat']);
else
  fnames.path = output_path;
  if strcmpi(fext, '.MAT'), fn = [fn,'mat']; end
  fnames.raw_file = fullfile(output_path, [fn, file_suffix, '.mat']);
  fnames.p_file = fullfile(output_path, ['p', fn, file_suffix, '.mat']);
  fnames.prj_file = fullfile(output_path, ['prj_', fn, file_suffix, '.mat']);
  fnames.prj_rawfile = fullfile(output_path, ['raw_', fn, file_suffix, '.mat']);
end