% function [s,X, dicompars]=read_dicom_image_file(name)
function [s,X, dicompars]=read_dicom_image_file(name)

dicompars = dicominfo(name, 'UseDictionaryVR', true);

% [all_names_sorted] = sort(all_names);
if isfield(dicompars, 'Manufacturer') && ...
    contains(dicompars.Manufacturer, 'Agilent') || ...
    contains(dicompars.Manufacturer, 'Bruker') 
        [s, X, dicompars] = default_loader(name);
      return
end

[fpath, ~, fext] = fileparts(name);
if isempty(fext), fext = '.'; end
all_files = dir(fullfile(fpath, ['*', fext]));
dirFlags = [all_files.isdir];
files_only = all_files(~dirFlags);
all_names = {files_only.name};

% exclude DICOMDIR
idx = cellfun(@(x) not(contains(x, 'DICOMDIR')),all_names);
all_names = all_names(idx);

StationName = safeget(dicompars, 'StationName', '');
if contains(StationName, 'PHILIPS-IJI1EMU')
  DICOMfrom = 'MRI3T';
elseif contains(StationName, 'XRAD-UofC')
      DICOMfrom = 'IMRT_Chuck';
elseif contains(StationName, 'XRAD')
      [s, X, dicompars] = default_loader(name);
      return;
elseif isfield(dicompars, 'Manufacturer')
  switch dicompars.Manufacturer
    case 'Molecubes NV'
      DICOMfrom = 'generic-processed';
    case 'Precision X-Ray'
      DICOMfrom = 'IMRT_Chuck';
    otherwise
      DICOMfrom = 'USI';
  end
else
  DICOMfrom = 'IMRT_Chuck';
end

switch DICOMfrom
  case 'IMRT_Chuck'
    [fpath, fname, fext] = fileparts(name);
    
    pos = strfind(fname, '_');
    if isempty(pos), return; end
          
    n1 = double(safeget(dicompars, 'Rows', 1));
    n2 = double(safeget(dicompars, 'Columns', 1));
    nSlice = safeget(dicompars, 'ImagesInAcquisition', 1);
    res12 = safeget(dicompars, 'PixelSpacing', 1);
    resS = safeget(dicompars, 'SliceThickness', 1);
    
    if exist(fullfile(fpath, sprintf([fname(1:pos(end)),'%04i', fext], 0)), 'file')
      fname = [fname(1:pos(end)),'%04i', fext];
      range = (1:nSlice)-1;
    elseif exist(fullfile(fpath, sprintf([fname(1:pos(end)),'%i', fext], 1)), 'file')
      fname = [fname(1:pos(end)),'%i', fext];
      range = 1:nSlice;
    end

    X = zeros(n1, n2, nSlice);
    slice_pos = zeros(nSlice, 3);
    
    for ii = range
      fname_dicom = fullfile(fpath, sprintf(fname, ii));
      X(:,:,ii+1) = dicomread(fname_dicom);
      dicompars = dicominfo(fname_dicom);
      
      if isfield(dicompars, 'ImagePositionPatient')
        slice_pos(ii+1, :) = dicompars.ImagePositionPatient(:)';
      end
    end
    if mean(diff(slice_pos(:,3))) < 0
      X = flip(X, 3);
    end
    
    s.bbox = [1*res12(1),n1*res12(1);1*res12(2),n2*res12(2);1*resS,nSlice*resS];
    s.dims = [n1,n2,nSlice];
    s.origin = sum(s.bbox,2)/2;
    s.pixsize = [res12(1); res12(2); resS];
    s.pixel_to_world = hmatrix_translate([-1 -1 -1])*hmatrix_scale(s.pixsize'.*[1 1 1])*...
      hmatrix_translate(-diff(s.bbox', [], 1)/2);
  case 'MRI3T'
    X = dicomread(name);
    X = flip(squeeze(X(:,:,1,:)),1);
    
    h = figure(1);
    imagesc(sum(X, 3)); axis image
    title('Select the area of interest.');
    a = gca;
    k = waitforbuttonpress;
    point1 = a.CurrentPoint;    % button down detected
    finalRect = rbbox;          % return figure units
    point2 = a.CurrentPoint;    % button down detected
    close(h);
    
    point1 = fix(point1(1,1:2));            % extract x and y
    point2 = fix(point2(1,1:2));
    yy = sort([point1(1),point2(1)]);
    xx = sort([point1(2),point2(2)]);
    X = X(xx(1):xx(2), yy(1):yy(2), :);
    
    vx = 0.234375;
    a = inputdlg({'Resolution X';'Resolution Y'},'Resolution',1,{num2str(vx);num2str(vx)});
    if ~isempty(a)
      vx = str2double(a{1});
    end
    
    sl = dicompars.SpacingBetweenSlices;
    
    s.bbox = [1*vx,size(X,1)*vx; 1*vx,size(X,2)*vx; 1*sl,size(X,3)*sl];
    s.dims = size(X);
    s.origin = sum(s.bbox,2)/2;
    s.pixsize = [vx; vx; sl];
    s.pixel_to_world = hmatrix_translate([-1 -1 -1])*hmatrix_scale(s.pixsize'.*[1 1 1])*...
      hmatrix_translate(-diff(s.bbox', [], 1)/2);
  case 'generic-processed' % Molecubes
    info = dicominfo(name);
    SliceThickness = info.SliceThickness;
%     NumberOfFrames = info.NumberOfFrames;
    X = dicomread(name);
    X = double(squeeze(X));
    n = size(X);
    
    s.bbox = [SliceThickness,n(1)*SliceThickness;SliceThickness,n(2)*SliceThickness;SliceThickness,n(3)*SliceThickness];
    s.dims = n;
    s.origin = sum(s.bbox,2)/2;
    s.pixsize = [SliceThickness; SliceThickness; SliceThickness];
    s.pixel_to_world = hmatrix_translate([-1 -1 -1])*hmatrix_scale(s.pixsize'.*[1 1 1])*...
      hmatrix_translate(-diff(s.bbox', [], 1)/2);

  case 'USI'
    X = dicomread(name);
    X = flip(squeeze(X(:,:,1,:)),1);
    
    Regions = dicompars.SequenceOfUltrasoundRegions.Item_1;
    yy = [Regions.RegionLocationMinX0, Regions.RegionLocationMaxX1];
    xx = [Regions.RegionLocationMinY0, Regions.RegionLocationMaxY1];
    
%     h = figure;
%     imagesc(sum(X, 3)); axis image
%     title('Select the area of interest.');
%     a = gca;
%     k = waitforbuttonpress;
%     point1 = a.CurrentPoint;    % button down detected
%     finalRect = rbbox;          % return figure units
%     point2 = a.CurrentPoint;    % button down detected
%     close(h);
%     point1 = fix(point1(1,1:2));            % extract x and y
%     point2 = fix(point2(1,1:2));
%     yy = sort([point1(1),point2(1)]);
%     xx = sort([point1(2),point2(2)]);

    
    X = X(xx(1):xx(2), yy(1):yy(2), :);
    
    vx = dicompars.PixelSpacing(1);
%     vx = pars_out.PixelSpacing(1);
    dx = 25/246;
    
    a = inputdlg('Input step size', 'USI parameters', 1, {num2str(dx)});
    if ~isempty(a)
      dx = str2double(a{1});
    end
    
    s.bbox = [1*vx,size(X,1)*vx; 1*vx,size(X,2)*vx; 1*dx,size(X,3)*dx];
    s.dims = size(X);
    s.origin = sum(s.bbox,2)/2;
    s.pixsize = [vx; vx; dx];
    s.pixel_to_world = hmatrix_translate([-1 -1 -1])*hmatrix_scale(s.pixsize'.*[1 1 1])*...
      hmatrix_translate(-diff(s.bbox', [], 1)/2);

    % im.Size = size(im.a).*[vx,vx,sl];
    % ibGUI(im)
end


function [s, X, pars_out] = default_loader(filename)
dicompars = dicominfo(filename, 'UseDictionaryVR', true);
pars_out = dicompars;

[fpath, ~, fext] = fileparts(filename);
if isempty(fext), fext = '.'; end
all_files = dir(fullfile(fpath, ['*', fext]));
dirFlags = [all_files.isdir];
files_only = all_files(~dirFlags);
all_names = {files_only.name};

% image resolution
SliceThickness = dicompars.SliceThickness;
PixelSpacing = dicompars.PixelSpacing(:);

VS = [PixelSpacing; SliceThickness];

N = length(all_names);

% slice size and datatype
S = dicomread(filename);
sz = size(S);
tp = class(S);
% pre-allocate data
VT = zeros([sz N], tp);
X  = zeros([sz N]);
POS = zeros(N,2);
% load each slice and its properties
for i=1:N
    ifname = fullfile(fpath, all_names{i});
    %         fprintf('Reading %s\n', ifname);
    VT(:,:,i) = squeeze(dicomread(ifname));
    info = dicominfo(ifname, 'UseDictionaryVR', true);
    if isfield(info, 'ImagePositionPatient')
        POS(i,:) = [info.ImagePositionPatient(3) i];
    else
        POS(i,:) = [i i];
    end
    if isfield(info, 'SliceLocation')
      fprintf('%i: SliceLocation %f\n', i, info.SliceLocation);
    end
end
% resort the slices according to the image position
POS = sortrows(POS,1);
for i=1:N
    X(:,:,i) = VT(:,:,POS(i,2));
end

pars_out.ImagePositionPatient = []; % block use of position for DICOM with wrong position
s.bbox = [VS(1),sz(1)*VS(1);VS(2),sz(2)*VS(2);SliceThickness,N*SliceThickness];
s.dims = [sz(:)',N];
s.origin = sum(s.bbox,2)/2;
s.pixsize = VS;
s.pixel_to_world = hmatrix_translate([-1 -1 -1])*hmatrix_scale(s.pixsize'.*[1 1 1])*...
    hmatrix_translate(-transpose(diff(s.bbox, 1, 2))/2);
