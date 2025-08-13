% function [mat, mat_info]=epr_ReadCWImageFile(name_com, opt)
function [mat, raw_info]=epr_ReadCWImageFile(file_name, opt)

if ~exist('opt', 'var'), opt = []; end

mat = [];
mat_info = [];
mat_info.pars_out = [];

% read data from file
fid=fopen(file_name, 'r');
if fid < 0
  error(['Can not open ''',file_name, ''' file.']);
end

mat_info = epr_ReadCWImageHeader(fid, opt);
FBP = mat_info.data.FBP;

% Check the file type
if strcmpi(mat_info.data.Origin, 'LabView')
  while ~feof(fid)
    str = fgetl(fid);
    if ~isempty(strfind(str,'% Total Projections')), break; end
  end
  if feof(fid), return; end
  
  % This is the first index
  prj = 1;
  while ~feof(fid)
    str = fgetl(fid);    
    iplr = str2double(str(1:2));
    iaz = str2double(str(3:4));
    ispec = str2num(str(5:6));
    if iplr == 0 || iaz == 0 || ispec == 0, mat_info.gidx(prj) = 0;
    else
      mat_info.gidx(prj) = ispec + (iaz-1)*FBP.nSpec + (iplr-1)*FBP.nSpec*FBP.nAz;
    end
    proj_size=fscanf(fid,'%d'); fgetl(fid);
    fgetl(fid);
    if mod(FBP.scanMethod,2)==1 
      % odd scanMethod ~ up scan only
      f=fscanf(fid,'%f',proj_size); fgetl(fid);
      if (FBP.swDefCode==1 || FBP.swDefCode==2)
        %             f0 =zeros((nbins-proj_size)/2,1);
        %             f = [f0; f; f0];
      elseif FBP.swDefCode==3
        %             f1=zeros(1,nbins);
        % %            if iaz==8
        % %               fprintf('11\n'); keyboard;
        % %            end
        %             [nStart, nEnd] = ...
        %                 ComputeSWindex(ispec,nspec,nbins,fL1(iplr,iaz),fL2(iplr,iaz));
        %             f1(nStart:nStart+proj_size-1)=f;
        %             %keyboard;
        %             f=f1;
      end
      mat(:,prj)=f;
    else
      %         % even scanMethod ~ triangular scan
      %         proj_size=proj_size/2;
      %         f=zeros(1,proj_size);
      %         f=fscanf(fid,'%f',[proj_size]);
      %         mat(:,ispec,iaz,iplr)=f;
      %         f=zeros(1,proj_size);
      %         f=fscanf(fid,'%f',[proj_size]);
      %         mat(:,nspec+1-ispec,iaz,iplr)=-f;
    end
    fgetl(fid); fgetl(fid);
    
    mat_info.mat_info(1:6,prj)=fscanf(fid,'%f',[6]); fgetl(fid);
    % number of bins zero-padded to be retrieved by unbin_mat_data
    mat_info.mat_info(7,prj)= 0; % nbins-proj_size;
    fgetl(fid);  fgetl(fid);
    
    
    [t1,t2] = ReadLabviewTimeStamp(fgetl(fid));
    mat_info.mat_info(8,prj)=t1;
    mat_info.mat_info(9,prj)=t2;
    
    prj = prj + 1;
    fgetl(fid);    
  end
else
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Modula %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  while 1
    str = fgetl(fid);
    if ~isempty(strfind(str,'010101')), break; end
  end
  
  % This is the first index
  prj = 1;
  while ~feof(fid)
    iplr = str2num(str(1:2));
    iaz = str2num(str(3:4));
    ispec = str2num(str(5:6));
    mat_info.gidx(prj) = (ispec-1) + (iaz-1)*nspec + (iplr-1)*nspec*naz;
    proj_size=fscanf(fid,'%d', 1); fgetl(fid);
    if mod(scanMethod,2)==1
      % odd scanMethod ~ up scan only
      f=zeros(1,proj_size);
      sc = 0; f1 = 1;
      f=fscanf(fid,'%f',[proj_size]);
      if (swDefCode==1 | swDefCode==2)
        %         f0 =zeros((nbins-proj_size)/2,1);
        %         f = [f0; f; f0];
      elseif swDefCode==3
        %         f1=zeros(1,nbins);
        %         %            if iaz==8
        %         %               fprintf('11\n'); keyboard;
        %         %            end
        %         [nStart, nEnd] = ...
        %           ComputeSWindex(ispec,nspec,nbins,fL1(iplr,iaz),fL2(iplr,iaz));
        %         f1(nStart:nStart+proj_size-1)=f;
        %         %keyboard;
        %         f=f1;
      end
      mat(:,prj)=f;
    else
      % even scanMethod ~ triangular scan
      %       proj_size=proj_size/2;
      %       f=zeros(1,proj_size);
      %       f=fscanf(fid,'%f',[proj_size]);
      %       mat(:,ispec,iaz,iplr)=f;
      %       f=zeros(1,proj_size);
      %       f=fscanf(fid,'%f',[proj_size]);
      %       mat(:,nspec+1-ispec,iaz,iplr)=-f;
    end
    mat_info.mat_info(1:6,prj)=fscanf(fid,'%f',[6]); fgetl(fid);
    % number of bins zero-padded to be retrieved by unbin_mat_data
    mat_info.mat_info(7,prj)= 0; % nbins-proj_size;
    prj_time=fscanf(fid,'%f',[6]); fgetl(fid);
    if isempty(prj_time)
      disp('Acquisition time data are not availaible. Old data...');
      return;
    end
    
    st=mod2mat_time(prj_time(1:3));
    fn=mod2mat_time(prj_time(4:6));
    if prj == 1, mat_info.time_start = st'; end
    mat_info.mat_info(8,prj)=etime(st', mat_info.time_start);
    mat_info.mat_info(9,prj)=etime(fn', mat_info.time_start);
    
    fgetl(fid); % One more time frequency
    
    prj = prj + 1;
    fgetl(fid);
    
    str = fgetl(fid);
  end
end
fclose(fid);

% remove repeated indexes
raw_info  =  epr_CopyFields([], mat_info, {'data', 'ModFrequency', 'ModAmplitude', 'ampHH'});
raw_info.nTrace = length(mat_info.gidx);
% tidx = -1*ones(1, FBP.nPolar*FBP.nAz*FBP.nSpec);
% gidx = mat_info.gidx(mat_info.gidx > 0);
% sidx = 1:length(gidx);
% tidx(gidx) = sidx;
% idx = tidx(tidx > -1); idx = sort(idx);
% mat_info.mat_info = mat_info.mat_info(:,idx);
% mat_info.gidx     = mat_info.gidx(idx);
% mat               = mat(:,idx);

% simplified version of normalization on scans number 'unbinning'
mat_info.service_fld = fix(mat/1E6);
scans = abs(mat_info.service_fld); scans(scans==0)=1;
mat = (mat - 1E6 * mat_info.service_fld)./scans;
mat(mat_info.service_fld == 0) = 0;

% bring experiment start time to 0
time_start = mat_info.mat_info(8,1);
mat_info.mat_info(8,:)=mat_info.mat_info(8,:) - time_start;
mat_info.mat_info(9,:)=mat_info.mat_info(9,:) - time_start;
% if experiment span over two days
idx = mat_info.mat_info(8,:) < 0; mat_info.mat_info(8,idx) = mat_info.mat_info(8,idx) + 24*60*60;
idx = mat_info.mat_info(9,:) < 0; mat_info.mat_info(9,idx) = mat_info.mat_info(9,idx) + 24*60*60;

FBP = raw_info.data.FBP;
raw_info.Dim   = [FBP.nSpec, FBP.nAz, FBP.nPolar];
raw_info.nP    = length(find(mat_info.gidx > 0));
raw_info.GradX = mat_info.mat_info(1,:)';
raw_info.GradY = mat_info.mat_info(2,:)';
raw_info.GradZ = mat_info.mat_info(3,:)';
raw_info.gidx = mat_info.gidx(:);
raw_info.data.FBP.MaxGradient = max(sqrt(raw_info.GradX.^2+raw_info.GradY.^2+raw_info.GradZ.^2));
raw_info.FieldSweep = mat_info.mat_info(5,:)';

%--------------------------------------------------------------------------
function mat_tv=mod2mat_time(mod_tv)

% input mod_tv (modula time vector) and converts
% to the Matlab time vector, mat_tv
% mod_tv = [day minute millisec][i][j]..[k]
%			day : bits 0-4 = day of month
%					:	bits 5-8 = month of year
%					: bits 9-15 = year-1900 (works through 2027)
%			month (1..12)
%    millisec = second*1000 + milliseconds
%
% mat_tv = [year month day hour minute seconds][i][j]..[k]

mod_tv_size=size(mod_tv);
dims=ndims(mod_tv);
n=prod(mod_tv_size(2:dims));
mod_tv=reshape(mod_tv,3,n);
for i=1:n
  if mod_tv(1,i)>0
    binday=dec2bin(mod_tv(1,i));
  else
    binday=dec2bin(2^16+mod_tv(1,i));
  end
  
  mat_tv(1,i)=bin2dec(binday(1:7))+1900;
  mat_tv(2,i)=bin2dec(binday(8:11));
  mat_tv(3,i)=bin2dec(binday(12:16));
  
  mat_tv(4,i)=floor(mod_tv(2,i)/60);
  mat_tv(5,i)=rem(mod_tv(2,i),60);
  
  if mod_tv(3,i)>0
    mat_tv(6,i)=mod_tv(3,i)/1000;
  else
    mat_tv(6,i)=(2^16+mod_tv(3,i))/1000;
  end
end
mat_tv=reshape(mat_tv,[6 mod_tv_size(2:dims)]);
% END OF function mat_tv=mod2mat_time(mod_tv);