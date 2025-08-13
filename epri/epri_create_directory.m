% td_CreateDirectory(directory_name)

function epri_create_directory(fpath)

if exist(fpath, 'dir'), return; end

sdir = strsplit(fpath, filesep);

% eliminate files
if strfind(sdir{end}, '.'), sdir = sdir(1:end-1); end
% eliminate empty entries
if isempty(sdir{end}), sdir = sdir(1:end-1); end
% hey this is mac
if isempty(sdir{1}), sdir{1}=[filesep]; end

mkpath = sdir{1};
for ii=2:length(sdir)
  mkpath = [mkpath, filesep, strtrim(sdir{ii})]; %#ok<AGROW>
  if ~exist(mkpath, 'dir')
    mkdir(mkpath);
  end
end