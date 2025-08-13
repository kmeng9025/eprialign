function varargout = PulseGradTableGUI(varargin)
% PULSEGRADTABLEGUI M-file for PulseGradTableGUI.fig
%      PULSEGRADTABLEGUI, by itself, creates a new PULSEGRADTABLEGUI or
%      raises the existing
%      singleton*.
%
%      H = PULSEGRADTABLEGUI returns the handle to a new PULSEGRADTABLEGUI or the handle to
%      the existing singleton*.
%
%      PULSEGRADTABLEGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PULSEGRADTABLEGUI.M with the given input arguments.
%
%      PULSEGRADTABLEGUI('Property','Value',...) creates a new PULSEGRADTABLEGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before PulseGradTableGUI_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to PulseGradTableGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help PulseGradTableGUI

% Last Modified by GUIDE v2.5 26-Feb-2008 15:33:24

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @PulseGradTableGUI_OpeningFcn, ...
  'gui_OutputFcn',  @PulseGradTableGUI_OutputFcn, ...
  'gui_LayoutFcn',  [] , ...
  'gui_Callback',   []);
if nargin && ischar(varargin{1})
  gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
  [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
  gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

%--------------------------------------------------------------------------
function PulseGradTableGUI_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
handles.CurrentDirectory = pwd;
handles.opt.RSMaxFreq = 4590;
handles.opt.dB = 1.024;
handles.opt.dL = -1;
handles.opt.N = 1;

guidata(hObject, handles);

%--------------------------------------------------------------------------
function varargout = PulseGradTableGUI_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

%--------------------------------------------------------------------------
function [image_type_code, ext] = SPIGetPars(handles)

image_type_code = '';
ext = [];

% number of steps fro uniform acquizition or Z steps
steps = str2num(handles.editSPISteps.String);
if length(steps) == 1
  image_type_code = [image_type_code, 'SP',num2str(steps)];
elseif length(steps) == 3
  image_type_code = [image_type_code, 'SP',num2str(steps(3)),'X',num2str(steps(1)),'Y',num2str(steps(2))];
  else
  image_type_code = [image_type_code, 'SP',num2str(max(steps))];
end

switch handles.pmSPIMethod.Value
  case 1, image_type_code = [image_type_code, 'S1'];
  case 2, image_type_code = [image_type_code, 'S2'];
  case 3, image_type_code = [image_type_code, 'S3'];
end

% maximum gradient
image_type_code = [image_type_code, 'G', get(handles.editSPIGrad, 'String')];

% baseline
baseline = [];
split_baseline = false;
bl_field_pc = eval(get(handles.eSplitOffset, 'String'));
bl_n = str2double(get(handles.eBLnProjections, 'String'));
switch get(handles.pmBLAlgorithm, 'Value')
  case 2, baseline = 'B-1'; %'before';
  case 3, baseline = 'B-2'; %'after';
  case 4, baseline = 'B-3'; %'before_after';
  case 5, baseline = ['B',num2str(bl_n)];
  case 6, split_baseline = true;
  otherwise, baseline = '';
end
image_type_code = [image_type_code, baseline];

% coordinate pole
switch get(handles.pmCoordPole, 'Value')
  case 1, CoordPole = 1; %'X';
  case 2, CoordPole = 2; %'Y';
  case 3, CoordPole = 3; %'Z';
end
image_type_code = [image_type_code, 'P', num2str(CoordPole)];

% zero gradient type
switch get(handles.pmZGAlgorithm, 'Value')
  case 2, zero_gradient = 'Z-1'; %'before';
  case 3, zero_gradient = 'Z-2'; %'after';
  case 4, zero_gradient = 'Z-3'; %'before_after';
  case 5, zero_gradient = ['Z', get(handles.eZGnProjections, 'String')];
  otherwise, zero_gradient = 'Z0';
end
image_type_code = [image_type_code, zero_gradient];

% order
% image_type_code = [image_type_code, msps_code];

%--------------------------------------------------------------------------
function [image_type_code, ext] = FBPGetPars(handles)

image_type_code = '';
ext = [];

switch get(handles.pmAlg, 'value')
  case 1, image_type_code = [image_type_code, 'SB']; ext.B0algorithm='SB';
  case 2, image_type_code = [image_type_code, 'MB']; ext.B0algorithm='MB';
  case 3, image_type_code = [image_type_code, 'RS']; ext.B0algorithm='RS';
end

% angle scheme
image_type_code = [image_type_code, get(handles.editFBPAngle, 'String')];
imtype = get(handles.pmImageType, 'Value') - 1;

if(imtype >= 1 && imtype <= 7)
  image_type_code = [image_type_code, 'A', get(handles.editFBPSpecAngle, 'String')];
end
image_type_code = [image_type_code, 'T', num2str(imtype)];

% maximum gradient
image_type_code = [image_type_code, 'G', get(handles.editFBPGrad, 'String')];

% image sampling type
switch get(handles.pmFBPMethod, 'Value')
  case 1, angle_sampling = 1; ext.minusG = false; % 'uniform_spatial_flip';
  case 2, angle_sampling = 2; ext.minusG = false; % 'uniform_angular_flip';
  case 3, angle_sampling = 1; ext.minusG = true; % 'uniform_spatial_flip';
  case 4, angle_sampling = 2; ext.minusG = true; % 'uniform_angular_flip';
  case 5, angle_sampling = 5; ext.minusG = false; %'Gage option N1';
  case 6, angle_sampling  =6; ext.minusG = false; % Equal Solid Angle 8Q
  otherwise
    angle_sampling = 1;
end
image_type_code = [image_type_code, 'S', num2str(angle_sampling)];

% baseline
baseline = [];
split_baseline = false;
bl_field_pc = eval(get(handles.eSplitOffset, 'String'));
bl_n = str2double(get(handles.eBLnProjections, 'String'));
switch get(handles.pmBLAlgorithm, 'Value')
  case 2, baseline = 'B-1'; %'before';
  case 3, baseline = 'B-2'; %'after';
  case 4, baseline = 'B-3'; %'before_after';
  case 5, baseline = ['B',num2str(bl_n)];
  case 6, split_baseline = true;
  otherwise, baseline = '';
end
image_type_code = [image_type_code, baseline];

% coordinate pole
switch get(handles.pmCoordPole, 'Value')
  case 1, CoordPole = 1; %'X';
  case 2, CoordPole = 2; %'Y';
  case 3, CoordPole = 3; %'Z';
end
image_type_code = [image_type_code, 'P', num2str(CoordPole)];

% zero gradient type
switch get(handles.pmZGAlgorithm, 'Value')
  case 2, zero_gradient = 'Z-1'; %'before';
  case 3, zero_gradient = 'Z-2'; %'after';
  case 4, zero_gradient = 'Z-3'; %'before_after';
  case 5, zero_gradient = ['Z', get(handles.eZGnProjections, 'String')];
  otherwise, zero_gradient = 'Z0';
end
image_type_code = [image_type_code, zero_gradient];

% order
switch get(handles.pmMSPS, 'Value')
  case 2 
    msps_code = 'O1'; 
  otherwise, msps_code = 'O0';
end
image_type_code = [image_type_code, msps_code];

% navigator
switch get(handles.pmNavigator, 'Value')
  case 2 
    nav_code = 'N1'; % For all delays
    nav_code = [nav_code, 'NN', num2str(str2double(get(handles.eNavNumber, 'String')))];
  otherwise, nav_code = 'N0';
end
image_type_code = [image_type_code, nav_code];

% MULTY B
if isequal(ext.B0algorithm, 'MB')
  image_type_code = [image_type_code, 'MA1'];
  Offsets = eval(get(handles.eMBFieldOffset, 'string'));
  image_type_code = [image_type_code, 'MS', num2str(length(Offsets))];
  image_type_code = [image_type_code, 'MF', num2str(max(Offsets))];
  
  %       case 'MA', MBpars.MBadaptive = fix(str2double(k(ii).val));
  %       case 'MS', MBsteps = fix(str2double(k(ii).val));
  %       case 'MF', Fmax = str2double(k(ii).val);
  %         MBpars.Offsets = -Fmax:2*Fmax/(MBsteps-1):Fmax;
end

if isequal(ext.B0algorithm, 'RS') && split_baseline
  image_type_code = [image_type_code, 'MA0'];
  image_type_code = [image_type_code, 'MS', num2str(length(bl_field_pc))];
  image_type_code = [image_type_code, 'MF', num2str(max(bl_field_pc))];
end

ext.bl_field_pc = str2double(get(handles.eSplitOffset, 'String'));
ext.Baseline_Offset = str2num(get(handles.eBLFieldOffset, 'String'));
ext.RandomB0 = str2double(get(handles.eRandomB0, 'String'));

%--------------------------------------------------------------------------
function pbCWpars_Callback(hObject, eventdata, handles)

prompt={'Enter the Rapid Scan top frequency [Hz]:',...
  'Enter the Field support [G]:', ...
  'Enter the Object support [cm]:', ...
  'Enter number of averages', ...
  'Number of cycles (for RS)', ...
  'Minimum sweep (for RS)', ...
  'Maximum sweep (for RS)', ...
  'Additional sweep (for RS)'};
defaultanswer={num2str(safeget(handles.opt, 'RSMaxFreq', 4590)),...
  num2str(safeget(handles.opt, 'dB', 1.024)),...
  num2str(safeget(handles.opt, 'dL', 3.0)),...
  num2str(safeget(handles.opt, 'N', 1)), ...
  num2str(safeget(handles.opt, 'RS_Ncyc', 2)), ...
  num2str(safeget(handles.opt, 'SWmin', 1)), ...
  num2str(safeget(handles.opt, 'SWmax', 20)), ...
  num2str(safeget(handles.opt, 'SWadd', 0))};

answer=inputdlg(prompt,'Input CW and RS parameters',1,defaultanswer);
if ~isempty(answer)
  handles.opt.RSMaxFreq = str2double(answer{1});
  handles.opt.dB = str2double(answer{2});
  handles.opt.dL = str2double(answer{3});
  handles.opt.N = str2double(answer{4});
  handles.opt.RS_Ncyc = str2double(answer{5});
  handles.opt.SWmin = str2double(answer{6});
  handles.opt.SWmax = str2double(answer{7});
  handles.opt.SWadd = str2double(answer{8});
  if handles.opt.dL > 0
    [image_type_code, ext_opt] = FBPGetPars(handles);
    opt = epri_DecodeImageType(image_type_code);
    MaxGrad = handles.opt.dB/handles.opt.dL*max(tan(pi/2+pi/opt.nSpec/2-pi/opt.nSpec*(1:opt.nSpec)));
    set(handles.editFBPGrad, 'String', num2str(MaxGrad));
  end
  guidata(hObject, handles);
end

%--------------------------------------------------------------------------
function pbFBPGenerate_Callback(hObject, eventdata, handles)

[image_type_code, ext_opt] = FBPGetPars(handles);
opt = epri_DecodeImageType(image_type_code);
[pars, extended_pars] = iradon_FBPGradTable(opt);

switch(hObject)
  case handles.pbFBPGenerate
    fname = [image_type_code, '_(', num2str(pars.nTrace), ').dat'];
    [fname, directory_name] = uiputfile(...
      {'*.dat', 'Text file (*.dat)'; '*.*',  'All Files (*.*)'}, ...
      'Save template', fullfile(handles.CurrentDirectory, fname));
    if isequal(fname,0) || isequal(directory_name,0), return; end
    handles.CurrentDirectory = directory_name;
    guidata(handles.MainFigure, handles);
    fid = fopen(fullfile(directory_name, fname), 'w');
    fprintf(fid, 'Gx[G/cm] Gy[G/cm] Gz[G/cm] Sweep[G] SwFreq[xx] Shots[xx] Offset[G] Trace[s]\n');
end

if ext_opt.minusG
  pars.G = - pars.G;
end

BASELINE_INDEX = 0;

% projections with baseline
baseline_idx = pars.service_idx == BASELINE_INDEX;

pars.swFraction = ones(pars.nTrace, 1); %*(1 + 2*ext_opt.OffsetB0/pars.deltaH);
%   swFractionBL = 2*ext_opt.OffsetB0/pars.deltaH;
%   pars.swFraction(baseline_idx) = swFractionBL;

if strcmp(opt.scheme, 'rapid_scan')
else
  pars.Offset = zeros(pars.nTrace, 1);
  if any(ext_opt.RandomB0 > 0.001)
    pars.Offset = pars.Offset + (rand(size(pars.Offset))-0.5)*ext_opt.RandomB0;
  end
  pars.Offset(baseline_idx) = ext_opt.Baseline_Offset;
end

% this is good for non adaptive protocol
if isequal(ext_opt.B0algorithm, 'MB')
  Offset = repmat(opt.MB0.Offsets(:), 1, pars.nP);
  pars.Offset(~baseline_idx) = Offset(:);
end

%  repeat experiment multiple times
repeat = str2num(get(handles.eRepeatExperiment, 'String'));
if repeat > 1
  tmp_pars = pars;
  for ii=2:repeat
    pars.G = [pars.G; tmp_pars.G];
    pars.swFraction = [pars.swFraction; tmp_pars.swFraction];
    pars.nP = pars.nP + tmp_pars.nP;
    pars.nTrace = pars.nTrace + tmp_pars.nTrace;
    pars.pidx.i = [pars.pidx.i; tmp_pars.pidx.i];
    pars.pidx.j = [pars.pidx.j; tmp_pars.pidx.j];
    pars.pidx.k = [pars.pidx.k; tmp_pars.pidx.k];
  end
end

pars.pidx = extended_pars;
pars.deltaH = handles.opt.dB;
dBdL = opt.MaxGradient/tan(max(extended_pars.alpha(:)));
pars.deltaL =  pars.deltaH/dBdL;
pars.N = handles.opt.N;
RS_Ncyc = safeget(handles.opt, 'RS_Ncyc', 1);

if pars.nSpec==1
  pars.Sweep = handles.opt.dB * ones(pars.nTrace, 1);
else
  pars.Sweep=handles.opt.dB*sqrt(2)*pars.UnitSweep;
  SWmax = safeget(handles.opt, 'SWmax', 20);
  SWmin = safeget(handles.opt, 'SWmin', 0);
  SWadd = safeget(handles.opt, 'SWadd', 20);
  pars.Sweep = min(pars.Sweep + SWadd , SWmax);
  pars.Sweep = max(pars.Sweep + SWadd , SWmin);
end
max_RS_SW_rate = min(pars.Sweep)*2*handles.opt.RSMaxFreq;
%   if isfield(pars, 'fsplit')
%     pars.Offset = pars.Offset+pars.fsplit .* abs(pars.Sweep);
%   end

if ~isfield(pars, 'Offset'), pars.Offset = zeros(pars.nTrace,1); end

idxP = pars.service_idx > 0;
idxB = pars.service_idx == 0;
idxG = pars.service_idx == -1;
set(handles.eLogWindow, 'String', sprintf('%i projections, %i baselines and %i zero-G (%i traces)\n are generated.', ...
  length(find(idxP)), length(find(idxB)), length(find(idxG)), pars.nTrace));

switch hObject
  case handles.pbFBPGenerate
    for ii=1:pars.nTrace
      if(hObject == handles.pbFBPGenerate)
        str = sprintf('%g, %g, %g, %g, %g, %g, %g, %g',...
          pars.G(ii,1), pars.G(ii,2), pars.G(ii,3),...
          pars.Sweep(ii), max_RS_SW_rate/pars.Sweep(ii)/2, ...
          fix(pars.N/(min(pars.Sweep)/pars.Sweep(ii))), ...
          pars.Offset(ii),...
          RS_Ncyc./(max_RS_SW_rate/pars.Sweep(ii)/2));
        fprintf(fid, [str, '\n']);
      end
    end
    fclose(fid);
    % CW export
    [~,fff] = fileparts(fname);
    pars.dummy_projections = str2double(handles.eDummyProjections.String);
    epri_WriteCWGradTable(fullfile(directory_name, [fff, '.txt']), pars, extended_pars);
    %
    fprintf('Gradient table was written to\n     ''%s''.\n',fullfile(directory_name, fname));
  case handles.pbFBPTable
    f = figure('Position',[200 200 800 500]);
    cnames = {'Gx|[G/cm]','Gy|[G/cm]','Gz|[G/cm]','G|[G/cm]', 'Sweep|G', 'SwFreq|kHz', 'Scan', 'Offset|[G]', 'Window|[s]'};
    rnames = cell(1,pars.nTrace);
    for ii=1:pars.nTrace, rnames{ii} = sprintf('%i',ii); end
    dat = zeros(pars.nTrace, 6);
    dat(:, 1:3) = pars.G;
    dat(:, 4) = sqrt(sum(dat(:, 1:3).*dat(:, 1:3), 2));
    dat(:, 5:6) = [pars.Sweep, max_RS_SW_rate./pars.Sweep/2*1e-3];
    dat(:, 7) = fix(pars.N./(min(pars.Sweep)./pars.Sweep));
    dat(:, 8) = pars.Offset;
    dat(:, 9) = RS_Ncyc./(max_RS_SW_rate./pars.Sweep/2);
    
    uitable('Parent',f, 'units', 'normalized','Data',dat,'ColumnName',cnames,...
      'RowName',rnames,'Position',[.025 .025 .95 .95]);
  case handles.pbFBPGradMap
    figure(100);
    clf
    idxP = pars.service_idx > 0;
    plot3(pars.G(idxP, 1), pars.G(idxP, 2), pars.G(idxP, 3),'.-'); hold on
    
    idxB = pars.service_idx == 0;
    vis_offset = 1.01;
    plot3(pars.G(idxB, 1)*vis_offset, pars.G(idxB, 2)*vis_offset, pars.G(idxB, 3)*vis_offset,'r*'); hold on
    
    idxG = pars.service_idx == -1;
    vis_offset = 0.99;
    plot3(pars.G(idxG, 1)*vis_offset, pars.G(idxG, 2)*vis_offset, pars.G(idxG, 3)*vis_offset,'g*'); hold off
    
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal
end
%--------------------------------------------------------------------------
function slSPISteps_Callback(hObject, eventdata, handles)
shift = get(handles.slSPISteps, 'Value');
set(handles.slSPISteps, 'Value', 0);
val = str2num(get(handles.editSPISteps, 'String')) + 1 * shift;
set(handles.editSPISteps, 'String', num2str(val));
UpdateSPI_Callback(hObject, eventdata, handles)

%--------------------------------------------------------------------------
function pbSPIGenerate_Callback(hObject, eventdata, handles)

SPI = epri_DecodeImageType(handles.editSPITemplate.String);
pars = td_GetSPIGradientTable(SPI);

idxP = pars.service_idx > 0;
idxB = pars.service_idx == 0;
idxG = pars.service_idx == -1;

pars.swFraction = ones(pars.nTrace, 1);
swFraction = 2*str2num(get(handles.eBLFieldOffset, 'String'))/pars.deltaH;
pars.swFraction(idxB) = swFraction;

pars.Offset = zeros(pars.nTrace, 1);
Offset = str2double(get(handles.eBLFieldOffset, 'String'));
swFractionBL = 2*Offset/pars.deltaH;
pars.swFraction(idxB) = swFractionBL;
pars.Offset(idxB) = Offset;

switch hObject
  case handles.pbSPIGenerate
    fname = [handles.editSPITemplate.String, '_(', num2str(pars.nTrace), ').dat'];
    [fname, directory_name] = uiputfile(...
      {'*.txt', 'Text file (*.dat)'; '*.*',  'All Files (*.*)'}, ...
      'Save template', fullfile(handles.CurrentDirectory, fname));
    if ~isequal(fname,0) && ~isequal(directory_name,0)
      handles.CurrentDirectory = directory_name;
      guidata(handles.MainFigure, handles);
      
      %   WriteGradTable(fullfile(directory_name, fname), pars);
      fprintf('Gradient table was written to\n     ''%s''.\n',fullfile(directory_name, fname))
      
      normal_delay = 0;
      off_delay = 0;
      return_delay = 0;
      
      %   normal_delay = str2num(get(handles.eBLDelay, 'String'))*1E-3;
      %   off_delay = str2num(get(handles.eBLDelayOff, 'String'))*1E-3;
      %   return_delay = str2num(get(handles.eBLDelayOn, 'String'))*1E-3;
      delays = normal_delay * ones(size(idxB));
      
      delays(idxB) = off_delay;
      delays(logical([0;idxB(1:end-1)])) = return_delay;
      delays = [delays(2:end);normal_delay];
      %   fname = sprintf('delays_%gs_%gs_%gs_%d%s.dat', ...
      %     normal_delay, off_delay, return_delay, length(idxB), BLMethod);
      % %   fid = fopen(fullfile(directory_name, fname), 'w');
      %   for ii=1:length(idx); fprintf(fid, '%g s\n',delays(ii)); end
      %   fclose(fid);
      
      % delay [s], gradX, gradY, gradZ,
      fid = fopen(fullfile(directory_name, [fname,'.dat']), 'w');
      fprintf(fid, 'Gx[G/cm] Gy[G/cm] Gz[G/cm] Offset[G] delay[s] \n');
      for ii=1:length(idxB)
        fprintf(fid, '%g, %g, %g, %g, %g\n',...
          pars.G(ii,1), pars.G(ii,2), pars.G(ii,3),...
          pars.Offset(ii), delays(ii));
      end
      fclose(fid);
      
    end
  case handles.pbSPIGradMap
    figure(100);
    clf
    
    plot3(pars.G(idxP,1), pars.G(idxP,2), pars.G(idxP,3),'.-'); hold on
    plot3(pars.G(idxB,1), pars.G(idxB,2), pars.G(idxB,3),'r*'); hold off
    % Grad1D = linspace(-Grad, Grad, Steps);
    % Grad3Dx = Grad1D(ones(Steps,1), :, ones(Steps,1));
    % Grad3Dy = permute(Grad3Dx, [2,3,1]);
    % Grad3Dz = permute(Grad3Dx, [3,1,2]);
    %
    % for jj=1:length(pars.pidx.i)
    %     xx(jj) = Grad3Dx(pars.pidx.i(jj),pars.pidx.j(jj),pars.pidx.k(jj));
    %     yy(jj) = Grad3Dy(pars.pidx.i(jj),pars.pidx.j(jj),pars.pidx.k(jj));
    %     zz(jj) = Grad3Dz(pars.pidx.i(jj),pars.pidx.j(jj),pars.pidx.k(jj));
    % end
    % plot3(xx, yy, zz,'.-');
    
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal
  case handles.pbSPITable
    f = figure('Position',[200 200 800 500]);
    cnames = {'Gx|[G/cm]','Gy|[G/cm]','Gz|[G/cm]','G|[G/cm]', 'Offset|[G]'};
    rnames = cell(1,pars.nTrace);
    for ii=1:pars.nTrace, rnames{ii} = sprintf('%i',ii); end
    dat = zeros(pars.nTrace, 5);
    dat(:, 1:3) = pars.G;
    dat(:, 4) = sqrt(sum(dat(:, 1:3).*dat(:, 1:3), 2));
    dat(:, 5) = pars.Offset;
    
    uitable('Parent',f, 'units', 'normalized','Data',dat,'ColumnName',cnames,...
      'RowName',rnames,'Position',[.025 .025 .95 .95]);
end

set(handles.eLogWindow, 'String', sprintf('%i projections, %i baselines and %i zero-G (%i traces)\n are generated.\n delay for 3cm: %g us', ...
  length(find(idxP)), length(find(idxB)), length(find(idxG)), pars.nTrace,...
  max(SPI.nSteps)/(2*2.8*SPI.MaxGradient*3)));

%--------------------------------------------------------------------------
function pbLVConcatenate_Callback(hObject, eventdata, handles)
LabViewScriptConcatenateGUI

%--------------------------------------------------------------------------
function slFBP_Callback(hObject, eventdata, handles)

shift = get(hObject, 'Value');
set(hObject, 'Value', 0);

switch hObject
  case handles.slFBPSpecAngle, step = 1; hh = handles.editFBPSpecAngle;
  case handles.slFBPAngle, step = 1; hh = handles.editFBPAngle;
  case handles.slFBPGrad, step = 0.1; hh = handles.editFBPGrad;
  case handles.slSPISteps, step = 1; hh = handles.editSPISteps;
  case handles.slSPIGrad, step = 0.1; hh = handles.editSPIGrad;
end

val = str2double(get(hh, 'String')) + step * shift;
set(hh, 'String', num2str(val));

UpdateFBP_Callback(hObject, eventdata, handles)

%--------------------------------------------------------------------------
function pmBLAlgorithm_Callback(hObject, eventdata, handles)
onoff = {'off','on'};
bl = get(handles.pmBLAlgorithm, 'value');

set(handles.eBLnProjections, 'Enable', onoff{(bl == 5) + 1});
set(handles.eBLFieldOffset, 'Enable', onoff{(bl > 1 && bl < 6) + 1});
set(handles.eSplitOffset, 'Enable', onoff{(bl == 6) + 1});

UpdateFBP_Callback(hObject, eventdata, handles)

%--------------------------------------------------------------------------
function UpdateFBP_Callback(hObject, eventdata, handles)

image_type_code = FBPGetPars(handles);
set(handles.editFBPTemplate, 'string', image_type_code)
UpdateSPI_Callback(hObject, eventdata, handles);

%--------------------------------------------------------------------------
function UpdateSPI_Callback(hObject, eventdata, handles)
image_type_code = SPIGetPars(handles);
set(handles.editSPITemplate, 'string', image_type_code)
