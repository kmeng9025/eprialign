function [ax,y,dsc]=opus_read(varargin)
% OPUS FT-IR data file loader
%   [ax,y,dsc]=opus_read(filename)
%   Loads one selected data section (see format description below )
% Maded to be part of the KAZAN Viewer
% alsi 24.11.2005
%
% FORMAT description
%   format is terrible. original data has no X-axis
%   Structure of the file is following:
%    first comming difraction patern
%    second, FFT(baseline) cutted for some sertain region
%    and third FFT(data)-FFT(baseline)
%   If during experiment baseline is recordered (extention is usually *.0),
%   the third part is not present
%   Data is in the "float32" binary format. 
% HEADERS
%   Data sections are separated by headers, starting usually with "CSF" and end up
%   with "END". There is also some common header with description of
%   experimental setup Parameter, which I was able to find was "High/Low
%   folding limit" Headers have variable name in ascii code and values in
%   "float64" binary HFL/LFL - upper and lower frequency for X-axis, in
%   FFT(data) :-) 

filename = varargin{1};
if nargin>1, datapart = varargin{2}; else datapart = 2; end
warning off MATLAB:nonIntegerTruncatedInConversionToChar;
fid=fopen(filename,'r', 'ieee-le');

%ntotal = 83236;
fseek(fid, 0, 'eof');
ntotal = ftell(fid);

frewind(fid);
tmpstr=char(fread(fid,ntotal,'schar')).'; % whole file in one string
frewind(fid);
tmpdat=fread(fid,ntotal/4,'float32'); % whole file in one array
frewind(fid);
tmpint=fread(fid,ntotal/4,'int32'); % whole file in one array

hfl = strfind(tmpstr, 'HFL');
fseek(fid, hfl+7, 'bof');
HighFold = fread(fid, 1, 'float64');
lfl = strfind(tmpstr, 'LFL');
fseek(fid, lfl+7, 'bof');
LowFold = fread(fid, 1, 'float64');

fxv = strfind(tmpstr, 'FXV'); % first X point
lxv = strfind(tmpstr, 'LXV'); % last X point
npts = strfind(tmpstr, 'NPT'); % Number of data points 
for ii = 1:length(fxv)
        fseek(fid, fxv(ii)+7, 'bof');    
    FreqStPoint(ii) = fread(fid, 1, 'float64');
        fseek(fid, lxv(ii)+7, 'bof');
    FreqEnPoint(ii) = fread(fid, 1, 'float64');
        fseek(fid, npts(ii)+7, 'bof');
    NDataPoints(ii) = fread(fid, 1, 'int32');
end
npts = strfind(tmpstr, 'RSN'); % Running sample number (size(data)?)
fseek(fid, npts+7, 'bof');
NPoints = fread(fid, 1, 'int32');

npts = strfind(tmpstr, 'ARS'); % number of bg. scans
fseek(fid, npts+7, 'bof');
NBGScans = fread(fid, 1, 'int32');
npts = strfind(tmpstr, 'ASS'); % number of sample scans
fseek(fid, npts+7, 'bof');
NScans = fread(fid, 1, 'int32');
% res = strfind(tmpstr, 'RES'); % Running sample number (size(data)?)
% fseek(fid, res+3, 'bof');
% Resol = fread(fid, 1, 'int32');

fclose(fid);

%stcom = strfind(tmpstr, 'CSF');
encom = strfind(tmpstr, 'END') + 3;

for ii = 1:length(fxv)
    tencom = encom(encom<fxv(ii));
    tencom = sort(tencom);
    stcom(ii) = tencom(end);
end

% 1-Interferogram, 2, 3 - FFT
if length(fxv)>2 % full set of datas
    datapart = min(length(fxv), datapart);
    if datapart==1,
        ax.xlabel = 'Mirror steps, pnts';
        ax.title = 'Interferogram';
    else
        ax.xlabel = 'Wavenumber, cm^{-1}';
        ax.title = 'FT-IR spectrum';
    end
elseif length(fxv)==2 % baseline
    % somehow in file with baseline first comming fft(baseline) and only
    % then interferogram (sado_maso inc. :)
    datapart = min(length(fxv), datapart);
    if datapart==2,
        ax.xlabel = 'Wavenumber, cm^{-1}';
        ax.title = 'FT-IR spectrum';
       datapart = 1;
    else
        ax.xlabel = 'Mirror steps, pnts';
        ax.title = 'Interferogram';
        datapart = 2;        
    end
end


y = tmpdat(floor((stcom(datapart)+3)/4)+1+(1 : NDataPoints(datapart)));

ax.x = linspace(FreqStPoint(datapart), FreqEnPoint(datapart), NDataPoints(datapart)).';
ax.y = 1;

ax.HFL = HighFold;
ax.LFL = LowFold;
dsc.HighFoldingLimit = [num2str(HighFold), ' cm-1'];
dsc.LowFoldingLimit = [num2str(LowFold), ' cm-1'];
dsc.NumberOfBGScans = [num2str(NBGScans)];
dsc.NumberOfSampleScans = [num2str(NScans)];
dsc.c1 = '???';