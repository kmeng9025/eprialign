function varargout = SpatialResolutionGUI(varargin)
% SPATIALRESOLUTIONGUI MATLAB code for SpatialResolutionGUI.fig
%      SPATIALRESOLUTIONGUI, by itself, creates a new SPATIALRESOLUTIONGUI or raises the existing
%      singleton*.
%
%      H = SPATIALRESOLUTIONGUI returns the handle to a new SPATIALRESOLUTIONGUI or the handle to
%      the existing singleton*.
%
%      SPATIALRESOLUTIONGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SPATIALRESOLUTIONGUI.M with the given input arguments.
%
%      SPATIALRESOLUTIONGUI('Property','Value',...) creates a new SPATIALRESOLUTIONGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SpatialResolutionGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SpatialResolutionGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SpatialResolutionGUI

% Last Modified by GUIDE v2.5 01-Oct-2011 10:35:38

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SpatialResolutionGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @SpatialResolutionGUI_OutputFcn, ...
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

% --------------------------------------------------------------------
function SpatialResolutionGUI_OpeningFcn(hObject, ~, handles, varargin)

% Choose default command line output for SpatialResolutionGUI
handles.output = hObject;
handles.image.data = double(varargin{1});
handles.image.data = handles.image.data / max(handles.image.data(:));
matLmm = mean(double(varargin{2}));

[~,handles.center_prj(1)] = max(sum(sum(handles.image.data, 2),3));
[~,handles.center_prj(2)] = max(sum(sum(handles.image.data, 1),3));
[~,handles.center_prj(3)] = max(sum(sum(handles.image.data, 1),2));

handles.image.ix = 1:size(handles.image.data, 1);
handles.image.iy = 1:size(handles.image.data, 2);
handles.image.iz = 1:size(handles.image.data, 3);
handles.image.x = linspace(-matLmm/2,matLmm/2,size(handles.image.data, 1));
handles.image.y = linspace(-matLmm/2,matLmm/2,size(handles.image.data, 2));
handles.image.z = linspace(-matLmm/2,matLmm/2,size(handles.image.data, 3));
handles.image.matLmm = matLmm;
handles.the_view = 1;

handles.setUp.ctrP = [0,0,5];

% Update handles structure
guidata(hObject, handles);
Draw(handles, 1);

% UIWAIT makes SpatialResolutionGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --------------------------------------------------------------------
function varargout = SpatialResolutionGUI_OutputFcn(~, ~, handles) 
varargout{1} = handles.output;

% --------------------------------------------------------------------
function Draw(handles, the_view)

opt = [];
rx = (1:size(handles.image.data, 1))';

cmx = squeeze(sum(sum(handles.image.data, 1),2))'; cmx = cmx - min(cmx);
cmy = squeeze(sum(sum(handles.image.data, 1),3))'; cmy = cmy - min(cmy);
cmz = squeeze(sum(sum(handles.image.data, 2),3))'; cmz = cmz - min(cmz);
center_prj =fix([sum(cmx(:).*rx)/sum(cmx), sum(cmy(:).*rx)/sum(cmy),  sum(cmz(:).*rx)/sum(cmz)]);

[yx, zx, yz]=epr_getslice3D(handles.image.data, center_prj, ...
    handles.image.ix, handles.image.iy, handles.image.iz);
pars = GetGUIPars(handles);
switch(the_view)
    case 1
        epr_DrawAxis(handles.axes1, handles.image.y, handles.image.x, yx, [], [], [], 'YX', opt);
        ticks = linspace(pars.axial(1), pars.axial(2), pars.n_axial);
        DrawAxialTicks(handles.axes1, handles.setUp.ctrP(1), ticks, pars.radius(1), pars.radius(2))
    case 2
        epr_DrawAxis(handles.axes1, handles.image.z, handles.image.x, zx, [], [], [], 'ZX', opt);
        theta = linspace(pars.theta(1), pars.theta(2), pars.n_rad) + 90;
        DrawRadialTicks(handles.axes1, handles.setUp.ctrP([2,1]), theta, pars.radius(1), pars.radius(2))
    case 3
        epr_DrawAxis(handles.axes1, handles.image.y, handles.image.z, yz, [], [], [], 'YZ', opt);
        ticks = linspace(pars.axial(1), pars.axial(2), pars.n_axial);
        DrawAxialTicks(handles.axes1, handles.setUp.ctrP(2), ticks, pars.radius(1), pars.radius(2))
end

% --------------------------------------------------------------------
function pars = GetGUIPars(handles)
try
    pars.theta(1) = str2double(get(handles.eAngStart, 'string'));
catch err
    pars.theta(1) = 0;
end
try
    pars.theta(2) = str2double(get(handles.eAngEnd, 'string'));
catch err
    pars.theta(2) = 0;
end
try
    pars.radius(1) = str2double(get(handles.eRadStart, 'string'));
catch err
    pars.radius(1) = 0;
end
try
    pars.radius(2) = str2double(get(handles.eRadEnd, 'string'));
catch err
    pars.radius(2) = 0;
end
try
    pars.n_rad = str2double(get(handles.eCircumference, 'string'));
catch err
    pars.n_rad = 7;
end

try
    pars.axial(1) = str2double(get(handles.eAxStart, 'string'));
catch err
    pars.axial(1) = -4;
end
try
    pars.axial(2) = str2double(get(handles.eAxEnd, 'string'));
catch err
    pars.axial(2) = 4;
end
try
    pars.n_axial = str2double(get(handles.eLength, 'string'));
catch perr
    pars.n_axial = 7;
end

% --------------------------------------------------------------------
function PutGUIPars(handles, pars)
set(handles.eAngStart, 'string', num2str(pars.theta(1)));
set(handles.eAngEnd, 'string', num2str(pars.theta(2)));
set(handles.eRadStart, 'string', num2str(pars.radius(1)));
set(handles.eRadEnd, 'string', num2str(pars.radius(2)));
set(handles.eCircumference, 'string', num2str(n_rad));
set(handles.eAxStart, 'string', num2str(pars.axial(1)));
set(handles.eAxEnd, 'string', num2str(pars.axial(2)));
set(handles.eLength, 'string', num2str(pars.n_axial));

% --------------------------------------------------------------------
function DrawRadialTicks(h, center, theta, r1, r2)
A0=center(2) + center(1)*1i;
theta = theta * pi / 180;
zStart=real(r1*exp(1i*theta)+A0);
zEnd=real(r2*exp(1i*theta)+A0);
xStart=imag(r1*exp(1i*theta)+A0);
xEnd=imag(r2*exp(1i*theta)+A0);
for ii = 1 : length(theta)
    plot([xStart(ii),xEnd(ii)],[zStart(ii),zEnd(ii)], ...
        'Linewidth', 2, 'color', [1,1,1],...
        'Parent',h)
end

% --------------------------------------------------------------------
function DrawAxialTicks(h, center, ticks, r1, r2)
for ii = 1 : length(ticks)
    plot(ticks(ii)*[1,1],[r1,r2] + center, ...
        'Linewidth', 2, 'color', [1,1,1],...
        'Parent',h)
        plot(ticks(ii)*[1,1],-[r1,r2] + center, ...
        'Linewidth', 2, 'color', [1,1,1],...
        'Parent',h)
end

% --------------------------------------------------------------------
function pViews_SelectionChangeFcn(hObject, ~, handles)
switch hObject
    case handles.rbView1, handles.the_view = 1;
    case handles.rbView2, handles.the_view = 2;
    case handles.rbView3, handles.the_view = 3;
end 
guidata(hObject, handles);
Draw(handles, handles.the_view);

% --------------------------------------------------------------------
function pbFindAxis_Callback(hObject, ~, handles)
set(handles.eMessages, 'string', ...
{'Click 6 points on the outer boundary of the phantom.';...
'These points will be used to estimate the exact center of the phantom.'});
axes(handles.axes1);
circPts=ginput(6);pause(0.5);
P=FindCircCenter(circPts);
handles.setUp.ctrP(1)=P(2); % EPR x
handles.setUp.ctrP(2)=P(1); % EPR z
handles.setUp.ctrP(3)=P(3); % radius of phantom
%handles.setUp.pts = GetRanges(r0,handles.image,handles.setUp.ctrP);
set(handles.eMessages, 'string', sprintf('Radius of phantom is %2.2f mm',P(3)));

guidata(hObject, handles);
Draw(handles, handles.the_view);

% --------------------------------------------------------------------
function eWinUpdate_Callback(~, ~, handles)
Draw(handles, handles.the_view);

% --------------------------------------------------------------------
function pbCalculate_Callback(~, ~, handles)
pars = GetGUIPars(handles);
        pars.all_theta = linspace(pars.theta(1), pars.theta(2), pars.n_rad);
        pars.all_axial = linspace(pars.axial(1), pars.axial(2), pars.n_axial);
[r, traces] = GetTraces(handles.image,handles.setUp, pars);
sigma = zeros(size(traces, 1), 1);
reducedChiSq = zeros(size(traces, 1), 1);
for ii=1:size(traces, 1)
    [sigma(ii), reducedChiSq(ii)] = FitProfileAuto(r, traces(ii,:), 1, handles.axes1);
    pause(.2)
end
pause(1)
sigma_cut = ChauvenetCut(sigma, -1);
SE = std(sigma_cut)/sqrt(length(sigma_cut)-1);
set(handles.eMessages, 'string', sprintf('Average FWHH is %5.2f mm (SE = %5.2f mm)',...
    mean(sigma_cut), SE)); % factor 2*sqrt(2*log(2)) removed
Draw(handles, handles.the_view);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function x=FindCircCenter(surfPts)
% function x=FindCirc(R, surfPts);
% find out the center of circle from
%   surfPts (surface points)
% it makes use of fminsearch
% surfPts(1,:)=[x1 y1], surfPts(2,:)=[x2 y2], and so on...
% number of surfPts can be variable, at least 3 pts are necessary.
% P(1),P(2) = center(x,y) of the circle
% P(3) = radius of the circle

ctr0=mean(surfPts);
r0=sqrt((surfPts(1,1)-ctr0(1,1))^2+(surfPts(1,2)-ctr0(1,2))^2);
[x]= fminsearch(@fCirc_all,[ctr0(1) ctr0(2) r0],optimset('MaxFunEvals',1000,'TolX',1e-8),surfPts);

% --------------------------------------------------------------------
function sqDev = fCirc_all(P,surfPts)
N=size(surfPts,1);
R=P(3);
fCirc = @(X,R,surfPt)sum(((surfPt - X)/R).^2);
sqDev=0;
for k=1:N
    sqDev = sqDev+(fCirc(P(1:2),R,surfPts(k,:))-1)^2;
end

% --------------------------------------------------------------------
function [r,traces] = GetTraces(image, setUp, pars)
% 3D case
A0=setUp.ctrP(2) + setUp.ctrP(1)*1i;
pars.all_theta = - (pars.all_theta) * pi / 180;
zStart=real(pars.radius(1)*exp(1i*pars.all_theta)+A0);
zEnd=real(pars.radius(2)*exp(1i*pars.all_theta)+A0);
xStart=imag(pars.radius(1)*exp(1i*pars.all_theta)+A0);
xEnd=imag(pars.radius(2)*exp(1i*pars.all_theta)+A0);
yStart = pars.all_axial;
yEnd = pars.all_axial;

npts = max(size(image.data));
r = linspace(pars.radius(1), pars.radius(2), npts);
kk = 1;

ntrace = length(pars.all_axial) * length(pars.all_theta);
vecx = zeros(ntrace, npts);
vecy = zeros(ntrace, npts);
vecz = zeros(ntrace, npts);
for jj=1:length(pars.all_axial)
    for ii = 1:length(pars.all_theta)
        vecx(kk,:) = linspace(xEnd(ii),xStart(ii), npts);
        vecy(kk,:) = linspace(yEnd(jj),yStart(jj), npts);
        vecz(kk,:) = linspace(zEnd(ii),zStart(ii), npts);
        kk = kk + 1;
    end
end

traces = interp3(image.y, image.x, image.z, image.data, ...
    vecy, vecx, vecz, 'spline');

% --------------------------------------------------------------------
function fwhm = PostSummary(fwhm, setUp, reducedChiSq)
nX=size(setUp.pts,2); 
if safeget(setUp, 'thldChiSq', 0) > 0
    subFWHM=zeros(size(fwhm.raw));
    idx=find(reducedChiSq<setUp.thldChiSq);
    subFWHM(idx)=fwhm.raw(idx);
else
    subFWHM=fwhm.raw;
end
for j=1:nX
    x=ChauvenetCut(subFWHM(j,:));
    if length(x)<2, x=fwhm.raw(j,:);end
    fwhm.N(j)=length(x);
    fwhm.avg(j)=mean(x);
    if fwhm.N(j)<2
        fwhm.SE(j)=0;
    else
        fwhm.SE(j)=std(x)/sqrt(fwhm.N(j)-1);
    end
end
fwhm.avgAll=mean(fwhm.avg);
fwhm.stDevAll=std(fwhm.avg);
fprintf('\n\n  FWHM = %1.2f +/- %1.2f (SD) mm\n',fwhm.avgAll,fwhm.stDevAll);
% END OF function PostSummary

% --------------------------------------------------------------------
function [sigma, reducedChiSq] = FitProfileAuto(x, y, sigSq, axH)
% function sigma = FitProfile(x, y, sigSq);
% x in mm
[fitVal]=myFitting(x,y);
xi=linspace(min(x),max(x),4*length(x));
yFiti = fitVal(1)*erf((xi-fitVal(2))/(sqrt(2)*fitVal(3)))+fitVal(4);
yFit = fitVal(1)*erf((x-fitVal(2))/(sqrt(2)*fitVal(3)))+fitVal(4);
sigma=abs(fitVal(3));
chiSq=sum((y-yFit).^2/sigSq);
reducedChiSq=chiSq/length(y);

cla(axH);
myFontSize=14;
plot(xi,yFiti,'r','LineWidth',2, 'Parent', axH);
plot(x,y,'.','MarkerSize',12, 'Parent', axH);
% axis([min(xi) max(xi) -0.2 1.2]);
axis tight;
xlabel('mm','FontSize',myFontSize);
ylabel('intensity','FontSize',myFontSize);

s1=sprintf('FWHM=%5.2f mm',2*sqrt(2*log(2))*sigma);
% s2=sprintf('\\chi^2/\\nu=%5.2f',reducedChiSq);
text(0.1,0.85,s1,'FontSize',myFontSize, 'units', 'normalized', 'Parent', axH);
%text(0.25*max(x),0.25*max(y),s2,'FontSize',myFontSize);
% END OF FitProfileAuto

% --------------------------------------------------------------------
function [s resnorm residual]=myFitting(x,y)
% s: fitted value
% s0: initial value
% s0(1): half amplitude of erf
% s0(2): center of erf (peak of 1st derivative)
% s0(3): sigma (~ 1 mm mostly)
% s0(4): DC shift ~ half amplitude of erf
if y(1)>y(length(y))
    s0(1)=-0.5*(max(y)-min(y));
else
    s0(1)=0.5*(max(y)-min(y));
end
jj=find(diff(y)==max(diff(y)));
s0(2)=x(jj(1)); % length(jj) can be greater than 1
s0(3)=1;
s0(4)=abs(s0(1));

% remove unnecessary parts
[mmax, max_idx] = max(y);
idx = y > mmax*0.9; idx(1:max_idx) = true;

options=optimset('display','off');
[s resnorm residual] = lsqcurvefit(@myFun, s0, x(idx), y(idx), [],[],options);

% --------------------------------------------------------------------
function y = myFun(s,x)
y= s(1)*erf((x-s(2))/(sqrt(2)*s(3)))+s(4);

% --------------------------------------------------------------------
function samplesOut=ChauvenetCut(samples,N)
% exclude outliers using Chauvenet's criterion
% N = # of repeated application of Chauvenet's cut
% if N=-1, repeat until there is no change

if nargin < 2, N=1; end
if length(samples) < 2
    samplesOut=samples;return
end

if N == -1
    samples1=ChauvenetCutSub(samples);
    n1=length(samples1); n=length(samples);
    while n1<n
        samples=samples1;
        samples1=ChauvenetCutSub(samples);
        n1=length(samples1); n=length(samples);
    end
else
    for i=1:N
        samples=ChauvenetCutSub(samples);
    end
end
samplesOut=samples;

% --------------------------------------------------------------------
function samplesOut=ChauvenetCutSub(samples)
n=length(samples);
m=median(samples);
s=std(samples);
try
X = norminv([0.5/n 1-0.5/n], m, s);
idx = find(samples>X(1) & samples<X(2));
samplesOut=samples(idx);
catch
 samplesOut = samples;   
end