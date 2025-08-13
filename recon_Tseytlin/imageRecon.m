function [axSpace,axSpect,Image,varargout]=imageRecon(Field,Spec,Par)
%% Rapid Scan Imaging Reconstruction Program Version 1.0
% Created on 24 April 2018 by Lukas Woodcock
% Adapted from scripts created by Mark Tseitlin
%Edited on 11 August by Kit Canny
%
% This program will take sets of gradient data and reconstruct them into a
% 2D Spectral-Spatial image
%
%% Syntax
%
% [axSpace,axSpect,Image]=ImageRS(Field,Spec,Par)
% [axSpace,axSpect,Image,slSpc,slNoGrad]=ImageRS(Field,Spec,Par)
% [axSpace,axSpect,Image,slSpc,slNoGrad,slSpa]=ImageRS(Field,Spec,Par)
%
%% Description
%
% [axSpace,axSpect,Image]=ImageRS(Field,Spec,Par)
%       will output the spatial and spectral axies as well as the
%       reconstructed image.
%
% [axSpace,axSpect,Image,slNoGrad]=ImageRS(Field,Spec,Par)
%       will output the spatial and spectral axies, the reconstructed
%       image and the 0 gradient value.
%
% [axSpace,axSpect,Image,slSpc,slNoGrad]=ImageRS(Field,Spec,Par)
%       will output the spatial and spectral axies, the reconstructed
%       image, a horizontal spectral slice from the image, and the
%       0 gradient value.
%
% [axSpace,axSpect,Image,slSpc,slNoGrad,slSpa]=ImageRS(Field,Spec,Par)
%       will output the spatial and spectral axies, the reconstructed
%       image, a horizontal spectral slice from the image, the deconvolved
%       0 gradient value, and a vertical spatial slice.
%
%% Input Arguments
%
% Field - Spectral axis in G.
% Spec - Spectral data.
% Par - Structure consisting of:

%       Par.sw:     The rapid scan sweep width in G.
%       Par.fwhm:   Full width-half height of narrowest line in G.
%       Par.grad:   Gradient values in G/cm
%       Par.method: Reconstruction Method, either 'tikh_0', 'tikh_1', or
%                   'pen-rose'
%       Par.tol:    Reconstruction Method Tolerance value
%       Par.harm:   Reconstruction Method Maxium Harmonic level
%       Par.slSpc:  Position in cm of Spectral Slice
%       Par.slSpa:  Position in G of Spatial Slice
%       Par.FOV:    Spatial Field of View
%       Par.cut:    Percent of projection to cut out
%       Par.app:    Percent Appodization
%       Par.pf:     Polynomial factor for bg removal


%% Input Variables

% Here the Par data is converted to isolated variables.

sw=Par.sw;          %sweep width
fwhm=Par.fwhm;      %full width half max
grad=Par.grad;      %gradient array
method=Par.method;       %reconstruction method

if isfield(Par,'slSpc')     %OPTIONAL
    SpecSlice=Par.slSpc;
end

if isfield(Par,'slSpa')     %OPTIONAL
    SpaSlice=Par.slSpa;
end

tol=Par.tol;        %spatial smoothing in high intensity
harm=Par.harm;      %image resolution between 1-100
cut=Par.cut;        %percent of spectral ends cut from 1-100
app=Par.app;        %apodization factor
PFfactor=Par.pf;

%% Calculated Values

slices=length(Spec(1,:));   %number of projections in a slice
MidSlice=(((slices-1)/2)+1);    %median value of projections

fov=Par.FOV;    % Field of view in [cm] , <= resonator size 
Gmax=grad(end);    % Maximum  Gradient [G/cm]
dz=fwhm/Gmax;      
zPoints=round(1*fov/dz);
axSpace=linspace(-fov/2,+fov/2,zPoints);

%% Other Information

Nh=length(Spec(:,1));%round(2*sw/fwhm); %number of pts in a projection
h=linspace(-sw/2,+sw/2,Nh);
Nh=length(h);           %length of 
Nz=length(axSpace);     %length of feild axis array
Ng=length(grad);        %length of the gradient array

%% Cut select section of each projection that is used to construct each image

pnt_cut=round(Nh*cut/100);   %number of pts cut
inx=pnt_cut:Nh-pnt_cut;      %indicies left after cut
CutSpec=Spec(inx,:);         %redefined Spec array (Y)
axSpect=Field(inx);          %redefined Feild array (X)
Nhc=length(axSpect);         %total number of pts in projection after cutting
pnt_app=round(Nhc*app/100);  %apodization of pts for smoothing   
jnx=pnt_app:Nhc-pnt_app;     %restructuring smoothed array
App=axSpect*0;               %Zero array of Feild 
App(jnx)=1;                  %populating Feild array
App=ConvSmooth(App',Nhc/1000);  %Populating feild array

%% BG POLY

axSpect=transpose(axSpect);
inj=abs(axSpect)>max(axSpect)*0.85;
if 1==1
    for n=1:slices
        pr=CutSpec(:,n);                                %renamed array of redefined spec
        ppp=polyfit(axSpect(inj),pr(inj),PFfactor);     %determining polynomial for background
        bgg=polyval(ppp,axSpect);      %background polynomial value
        pr=pr-bgg;                     %subtracts background polynomial value from spec array
        pr=zeroLine(pr,0.05);          %adjusts the new slice to a offset of 5%
        pr=pr.*App;                    %adjusts previous feild array to new offset and adjusted parameters
        CutSpec(:,n)=pr;               %redefines array name of Spec
    end
end

% Loop is inoperable and never accessed

% if 1==0             
%     bg_all=min(CutSpec')'; %#ok<UDIM>
%     ppp=polyfit(axSpect,bg_all,1);
%     bg_all=polyval(ppp,axSpect);
%     for n=1:slices
%         pr=CutSpec(:,n);
%         CutSpec(:,n)=pr-bg_all;
%     end
%      fig(CutSpec);
%     axSpect=transpose(axSpect);
% end

%%  RECONSTRUCTION

TT=zeros(Ng,Nz,Nhc);    %zeroing array to build image

for n=1:Ng          %number of gradients
    
    %Frequency domain method to populate array 
    
    [v,~]=fftM(axSpect,axSpect);    
    w=ifftshift(2*pi*v);   % w= DC ... wmax/w -wmax/2 ... -dw. converted to angular units, switched low and high frequency positions
    [W,Z]=meshgrid(w,axSpace);      
    T=exp(+1i*Z*grad(n).*W);  %bringing in gradient information
    TT(n,:,:)=T;           % f-domain shift matrix
end

PR=fft(CutSpec',[],2);      %Spectral information to apply to regulization methods

x_PH=zeros(Nz,Nhc);        %Zero array with length field axis and redefined field axis. Image will be reconstructed into this array. 

%% Regularization operators
    %in paper, first derivative tikhonov method does not work. Identity
    %method is similar to pseudo-inversion method.

v0=ones(1,Nz);
v1=ones(1,Nz-1);
D0=diag(v0,0);
D1=diag(v1,-1)+diag(-v1,+1);
DD1=D1'*D1;
DD0=D0'*D0;

%% Cases for Different Methods

fH=1; % first harmonic

switch method
    case 'pen-rose'  %pseudo-inversion
        for m=fH:harm
            L=TT(:,:,m);
            b=PR(:,m);
            xx=pinv(L,tol)*b;
            x_PH(:,m)=xx;
        end
        
    case 'tikh_0'   %identity method
        for m=fH:harm
            L=TT(:,:,m);
            b=PR(:,m);
            LL=L'*L;
            xx=(LL+tol*DD0)\L'*b;
            x_PH(:,m)=xx;
        end
        
    case 'tikh_1'   %first derivative method, found to not work well
        for m=fH:harm
            L=TT(:,:,m);
            b=PR(:,m);
            LL=L'*L;
            xx=(LL+tol*DD1)\L'*b;
            x_PH(:,m)=xx;
        end
end

%% Outputs

Image=real(ifft(x_PH,[],2));

if nargout>3
    if nargout==4
        varargout{1}=CutSpec(:,MidSlice);
    elseif nargout==5       %based on number of arguments in calling function
        SpectralSlice=find((axSpace-SpecSlice)>0,1);
        varargout{1}=Image(SpectralSlice,:);
        varargout{2}=CutSpec(:,MidSlice);
    elseif nargout==6
        SpectralSlice=find((axSpace-SpecSlice)>0,1);
        SpatialSlice=find((axSpect-SpaSlice)>0,1);
        varargout{1}=Image(SpectralSlice,:);
        varargout{2}=CutSpec(:,MidSlice);
        varargout{3}=Image(:,SpatialSlice);
    else
        error('Number of outputs doesnt match valid output format.');
    end    
end

end

function res=ConvSmooth(b,f)
%  function res=ConvSmooth(b,f);
%  res=b;f=5;
ss=size(b); s1=ss(1); s2=ss(2);
kern=lorenz1(s1,f);
res=b;
for i=1:s2
    tmp=b(:,i);
    yyy=myConv(tmp,kern',1);
    res(:,i)=yyy;
end
end

function res=myConv(data,kern,zp)
% zp - zeropadding
L=length(data);
ss=size(data);
x=zeros(ss);
x=[x; x];
if zp==0
    x=[];
end
if ss(1)==1
    data=[x data x];
    kern=[x kern x];
end

if ss(2)==1
    data=[x; data; x];
    kern=[x; kern; x];
end
% nnn=Round[FirstMom[data]];
tmp=fft(data);
tmp2=fft(fftshift(kern));
tmp3=(ifft(tmp.*tmp2));

if zp==0
    res=tmp3;
else
    res=tmp3(2*L+1:3*L);
end
end

function res=lorenz1(len,dx)
x=(1:len); 
y=1./((x-len/2).^2+ dx^2)/pi;
y=dx*y;
y=y/sum(y);
res=y;
end

function [res,tmp]=zeroLine(spectrum,extent)
%% Spectral Offset Removal
% calculates a ramp based on slope between first 5% and last 5% of array
% and subtracts from the array

L=length(spectrum);
edge=round(extent*L);
if edge==0
    edge=1;
end

Sleft=sum(spectrum(1:edge))/edge;
Sright=sum(spectrum(L-edge+1:L))/edge;
s=size(spectrum);

tmp=Sleft+(Sright-Sleft)/L*(1:L);
if size(tmp)~=s
    tmp=conj(tmp');
end

res=spectrum-tmp;
end