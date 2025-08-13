% kv_clinepr_read Read data in ClinEPR format
%
%   data = kv_clinepr_read(filename,...)
%   [ax,data] = kv_clinepr_read(filename,...);
%   [ax,data,dsc] = kv_clinepr_read(filename,...);

function varargout=kv_clinepr_read(filename, varargin)

opt = [];
ax = [];
y = [];
dsc = [];
if nargin > 1
    if ~mod(nargin-1,2)
        for kk=1:2:nargin-1
            opt=setfield(opt, lower(varargin{kk}), varargin{kk+1});    
        end
    else, error('Wrong amount of arguments')
    end
end
[pathstr, fname, ext] = fileparts(filename);

ax.nPoints = 0;

mode = 0; % header
header_index = 0;
data_index = 1;
fid = fopen(filename);
tline = fgetl(fid);
while ischar(tline)
  if mode == 0
    if contains(tline, ':')
      pos = strfind(tline, ':');
      fieldname = tline(1:pos-1);
      fieldname = fieldname(fieldname ~= ' ' & fieldname ~= '[' & fieldname ~= ']' & fieldname ~= '/' & fieldname ~= '-'  & fieldname ~= '*');
      dsc.(fieldname) = tline(pos+1:end);
    elseif contains(tline, 'No. samples/scan')  
      pos = strfind(tline, 'No. samples/scan');
      ax.nPoints = sscanf(tline(pos+length('No. samples/scan'):end), '%i');
    elseif contains(tline, 'Scan range/Gauss')  
      pos = strfind(tline, 'Scan range/Gauss');
      ax.ScanRange = sscanf(tline(pos+length('Scan range/Gauss'):end), '%i');
    elseif contains(tline, '***')
      mode = 1;
      y     = zeros(ax.nPoints,4);
      ax.x  = zeros(ax.nPoints,1);
    elseif ~isempty(tline)
      dsc.("header"+num2str(header_index)) = tline;
      header_index = header_index + 1;
    end
  elseif mode == 1
    yyyyy = sscanf(tline, '%g,%g,%g,%g,%g');
    ax.x(data_index) = yyyyy(1);
    y(data_index,:) = yyyyy(2:5);
    data_index = data_index + 1;
  end
  tline = fgetl(fid);
end
fclose(fid);

% assign output depending on number of output arguments
% requested
switch nargout
  case 1
    varargout = {spec};
  case 2
    varargout = {ax, y};
  case 3
    varargout = {ax, y, dsc};
end