% kv_read_halpern Read data in UC format
%
%   data = d01read(filename,...)
%   [ax,data] = d01read(filename,...);
%   [ax,data,dsc] = d01read(filename,...);

function varargout=kv_read_halpern(filename, varargin)

opt = [];
dsc = [];
if nargin > 1,
    if ~mod(nargin-1,2)
        for kk=1:2:nargin-1
            opt=setfield(opt, lower(varargin{kk}), varargin{kk+1});    
        end
    else, error('Wrong amount of arguments')
    end
else, error('Usage: kv_read_halpern(filename, src). Options: is2D.');
end
[pathstr, fname, ext] = fileparts(filename);

switch lower(ext)
  case {'.img'}
    [spec,mat_info]=epr_ReadCWImageFile(filename);
    ax.x = (1:size(spec, 1))';
    ax.y = (1:size(spec, 2))';
    dsc = mat_info;
  otherwise
    % type *.number
    slice = str2num(ext(2:end));
    if isempty(slice)
      % type *.Snumber
      slice = str2num(ext(3:end));
    end
    [pars_out,Z,range_out,prefix,title,comment]=...
      read_bin_spectrum(fullfile(pathstr,fname),slice,slice,1);
    spec = Z(:,2);
    ax.x = Z(:,1);
    ax.xlabel = 'Field,G';
end

% assign output depending on number of output arguments
% requested
switch nargout
  case 1,
    varargout = {spec};
  case 2,
    varargout = {ax, spec};
  case 3,
    varargout = {ax, spec, dsc};
end