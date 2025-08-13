% fpath = epr_GetIniPath
% fname = epr_GetIniPath(ini_file)
% returns the path for the ini file or ini file name (if provided) for different OS

function fname = epr_GetIniPath(ini_file)

if ispc
  [a, fpath] = dos('echo %USERPROFILE%');
  fpath = fullfile(strtrim(fpath), 'My Documents');
elseif isunix
  fpath = '~/Documents';
else
  fpath = fileparts(which(ini_file));
end
if exist('ini_file', 'var') && ~isempty(ini_file)
  fname = fullfile(strtrim(fpath), [ini_file,'.ini']);
else
  fname = strtrim(fpath);
end
