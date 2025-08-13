% kv_read_spinsight Read data from Spinsight
%
%   data = d01read(filename,...)
%   [ax,data] = d01read(filename,...);
%   [ax,data,dsc] = d01read(filename,...);

function varargout=kv_read_spinsight(filename, varargin)

opt = [];
if nargin > 1,
    if ~mod(nargin-1,2)
        for kk=1:2:nargin-1
            opt=setfield(opt, lower(varargin{kk}), varargin{kk+1});    
        end
    else, error('Wrong amount of arguments')
    end
end

[pathstr, fname, ext] = fileparts(filename);

ss = inimanage(fullfile(pathstr, 'acq'));

npoints = str2num(safeget(ss.noname, 'al', '0'));
[dw, unit, kk] = kvgetvalue(safeget(ss.noname, 'dw', '0'));
ax =   [];
ax.x = [1:npoints]'*dw;
ax.xlabel = ['Time, ',kk,unit];

dsc =  ss.noname;

fid=fopen(fullfile(pathstr, 'data'),'r', 'ieee-be');
if fid<1, error(['File ''',fname,''' can not be open for read.']);end

tmpdata=fread(fid,npoints*2,'float32');
fclose(fid);

data = tmpdata(1:npoints) + i*tmpdata(npoints+1:end);

% assign output depending on number of output arguments
% requested
switch nargout
case 1,
    varargout = {data};
case 2,
    varargout = {ax, data};
case 3,
    varargout = {ax, data, dsc};
end
