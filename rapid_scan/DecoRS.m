function [Field,varargout]=DecoRS(Time,Spec,Par,Type)
%% Rapid Scan Deconvolution Program Version 2.2
% Created on 3 April 2018 by Lukas Woodcock
% 
% This program will take any rapid scan data and deconvolve to recover the
% slow scan spectra.
%
%% Syntax
%
% [Field,Spec]=DecoRS(Time,Spec,Par,'Lin');
% [Field,RealCh,ImagCh]=DecoRS(Time,Spec,Par,'Lin');
% [Field,Ru,Iu,Rd,Id]=DecoRS(Time,Spec,Par,'Lin');
% [Field,Spec]=DecoRS(Time,Spec,Par,'Sin');
% [Field,Up,Down]=DecoRS(Time,Spec,Par,'Sin');
%
%% Description
%
% [Field,Spec]=DecoRS(Time,Spec,Par,'Lin')
%       will output the field axis as well as the summed final spectrum of
%       a linear dataset.
%
% [Field,RealCh,ImagCh]=DecoRS(Time,Spec,Par,'Lin')
%       will output the field axis as well as the Real channel spectrum and
%       the imaginary channel spectrum of a linear dataset.
%
% [Field,Ru,Iu,Rd,Id]=DecoRS(Time,Spec,Par,'Lin')
%       will output the field axis as well as the real up scan spectrum,
%       the imaginary up scan spectrum, the real down scan spectrum, and
%       the imaginary down scan spectrum of a linear dataset.
%
% [Field,Spec]=DecoRS(Time,Spec,Par,'Sin')
%       will output the field axis as well as the summed final spectrum of
%       a sinusoidal dataset.
%
% [Field,Up,Down]=DecoRS(Time,Spec,Par,'Sin')
%       will output the field axis as well as the Up scan spectrum and the
%       down scan spectrum of a sinusoidal dataset.
%
%% Input Arguments
%
% Time - Rapid Scan Time axis in ns.
% Spec - Rapid Scan Spectral data.
% Par  - Structure consisting of:
%        Par.cf:     The center field in G.
%        Par.sw:     The rapid scan sweep width in G.
%        Par.sf:     The rapid scan frequency in Hz.
%        Par.ph:     The phase correction in degrees. If unspecified will
%                    use a GUI.
%        Par.up:     The initial half cycle sweep direction; either 'up' or
%                    'down'. If unspecified will use a GUI.
%        Par.fp:     The rapid scan first point correction. If unspecified
%                    will use a GUI.
%        Par.fwhm:   For 'Sin' only; Full width-half height of narrowest
%                    line in G.
%        Par.Ortho:  OPTIONAL; signal channel orthoganality correction.
%        Par.Acorr:  OPTIONAL; signal channel amplitude correction.
%        Par.nH:     OPTIONAL; for 'Sin' only; number of background
%                    harmonics.
% Type - Data type; either 'Sin' for sinusoidal data or 'Lin' for linear.

%% Data Type Switch

switch Type
    case 'Lin'
        %% Run Deconvolution
        
        [Field,ru,iu,rd,id]=Lin(Time,Spec,Par);
        
        %% Combine Up and Down Scans
        % Up and Down scans are added together into the real and imaginary channel
        
        RealCh=ru+fliplr(rd);
        ImagCh=iu+fliplr(id);
        
        %% Final Spectrum
        % Real and Imaginary combined to a simngle trace
        
        Final=RealCh+ImagCh;
        
        %% Outputs
        % Depending on the number of outputs, either the Real and Imaginary
        % channel, the Seperated ups and downs, or a final spectrum is output
        
        if nargout~=0
            if nargout==2
                varargout{1}=Final;
            elseif nargout==3
                varargout{1}=RealCh;
                varargout{2}=ImagCh;
            elseif nargout==5
                varargout{1}=ru;
                varargout{2}=iu;
                varargout{3}=fliplr(rd);
                varargout{4}=fliplr(id);
            else
                error('Number of outputs doesnt match valid output format.');
            end
        end
        
    case 'Sin'
        %% Run Deconvolution
        
        [Field,Up,Down,fp,ph,up]=Sin(Time,Spec,Par);
        
        %% Final Spectrum
        % Up and Down combined to a simngle trace
        
        Final=Up+Down;
        
        %% Outputs
        % Depending on the number of outputs, either the Real and Imaginary
        % channel, the Seperated ups and downs, or a final spectrum is output
        
        if nargout~=0
            if nargout==2
                varargout{1}=Final;
            elseif nargout==3
                varargout{1}=Up;
                varargout{2}=Down;
            elseif nargout==5
                varargout{1}=Final;
                varargout{2}=fp;
                varargout{3}=ph;
                varargout{4}=up;
            else
                error('Number of outputs doesnt match valid output format.');
            end
        end
end

end

function [Field,ru,iu,rd,id]=Lin(Time,Spec,Par)
%% Linear Deconvolution Program 
% Created by Lukas Woodcock on 7 Sep 2017.
% Adapted from fortran code developed by Dr. Sandra Eaton.

%% Check for Proper Spectral and Time Axis Dimensions
% EPR Load often brings in the data as an Nx1 dataset. This Corrects the
% data to a 1xN dataset.

if length(Spec(:,1))>length(Spec(1,:))
    Spec=transpose(Spec);
end

if length(Time(:,1))>length(Time(1,:))
    Time=transpose(Time);
end

%% Input Variables
% Here the Par data is converted to isolated variables.

cf=Par.cf;      % Center Field in G
swpu=Par.sw;    % Sweep width in G
fswp=Par.sf;    % Scan freq. in Hz

%% Calculated Values
% Phase corrected spectral data, amplitude correction, the number of points
% in a cycle and half cycle, and the first point are calculated here. The
% real and imaginary channels are also seperated. The user can override
% defult amplitude calibrated instrument values are known. Corrections are
% applied to the raw data. If first point is not specified, a GUI to
% determine it is output before moving on.

% Amplitude Correction

if isfield(Par,'fsdd')==0
    if isfield(Par,'ACorr')==1
        ACorr=Par.ACorr;
    else
        ACorr=1;
    end
    Spec=complex(ACorr*real(Spec),imag(Spec));
end

% Time Base (ns)

tb=Time(1,2)-Time(1,1);
tb=tb*1e-9;         % Time Base in ns

% Cycle/Half Cycle Determination

NPC=1/(tb*fswp);        % Full Cycle Points
NPHc=round(NPC/2);      % Half Cycle Points
NPC=NPHc*2;             % Corrected Full Cycle Points
assignin('base','NPC',NPC);
% Determine First Point and Phase

if isfield(Par,'fp')==0 || isfield(Par,'ph')==0 || isfield(Par,'up')==0 
    [NPStart,ph,up]=findFPnPH(Time,Spec,Par,'Lin',NPC);
else
    NPStart=Par.fp; % First point correction
    ph=Par.ph;      % Phase correction in degrees
    up=Par.up;      % Up or Down Scan
end

Spec=Spec*exp(1i*ph/180*pi); % Phase Correction

%% Matrix Definition
% Loop matrices are declared here.

tempSpec=zeros(1,NPC);
ru=zeros(1,NPHc);
iu=zeros(1,NPHc);
rd=zeros(1,NPHc);
id=zeros(1,NPHc);
ic=complex(0,1);

%% Create x Axis
% Creates a Field axis in units of G

x1=cf-(swpu/2);
x2=cf+(swpu/2);
xstep=swpu/NPHc;

Field=x1:xstep:x2-xstep;

%% Average Full Cycles and Correct First Point
% Full cycles are combined into single, averaged cycle. The begining of the
% scan is corrected to index 1.

for q=1:NPC
    tempSpec(q)=mean(Spec(q:NPC:end));
end

tempSpec=circshift(tempSpec,[0,-NPStart]);

%% Orthoganality Correction
% The imaginary channel is orthganality corrected. Additionally, a hilbert
% transform is performed to extract the additional absorption data.

% Seperate Real and Imaginary Channel
tempr=real(tempSpec);
tempi=imag(tempSpec);

if isfield(Par,'fsdd')==0
    if isfield(Par,'Ortho')==1
        Ortho=Par.Ortho;
    else
        Ortho=0;
    end
    hilbDta=hilbert(tempi);
    hilbDta=hilbDta*exp(1i*Ortho/180*pi);
    tempi=imag(hilbDta);
end

tempr=tempr-tempr(1);
tempi=tempi-tempi(1);

%% Recombine Real and Imag Channels
% Here the real and imaginary channels are complexed as well as the array
% zero-filled to the number of points.

fdat=complex(tempr,tempi);

%% Initial Fourier Transform
% The data is converted from the time to frequency domain.

fdat=fft(fdat);

%% Driving function removal.
% The driving function is calculated and removed from the data.

fwid=(mt2mhz(swpu*2)*1e5);
delt=(1/fwid);
b=(mt2mhz(swpu)*1e5)*2*pi*fswp*2; % Rate Equation

cv=zeros(1,NPC);
cv(1:NPHc)=exp(-ic*0.5*b*((0:NPHc-1)*delt).^2);
cv(NPHc+1:NPC)=fliplr(conj(exp(-ic*0.5*b*((0:NPHc-1)*delt).^2)));
fdat=fdat./cv;

%% Inverse Fourier Transform
% The data is taken back to the time domain and seperated into real and
% imaginary channels.

fdat=ifft(fdat);

tempr=real(fdat);
tempi=imag(fdat);

%% Seperate Up and Down Scans
% Up and Down scans are seperated into two datasets. First dataset is
% determined by choosing 'up' or 'down'.

switch up
    case 'up'
        ru(1:NPHc)=tempr(1:NPHc);
        iu(1:NPHc)=tempi(1:NPHc);
        
        rd(1:NPHc)=tempr(NPHc+1:end);
        id(1:NPHc)=-tempi(NPHc+1:end);
    case 'down'
        rd(1:NPHc)=tempr(1:NPHc);
        id(1:NPHc)=-tempi(1:NPHc);
        
        ru(1:NPHc)=tempr(NPHc+1:end);
        iu(1:NPHc)=tempi(NPHc+1:end);
end

end

function [H,A,B,fp,ph,up]=Sin(Time,Spec,Par)
%% Sinusoidal Rapid Scan Deconvolution with Background Correction
% This program will take sinusoidal rapid scan data and deconvolve it to
% recover the slow scan spectum.
%
% Created by Mark Tseytlin (mark.tseytlin@du.edu)
% Updated 28 Mar 2018; Lukas Woodcock (lukas.woodcock@du.edu)
%
% References
%
% [1] Tseitlin et al. Deconvolution of sinusoidal rapid EPR scans.
% J. Mag. Res. 208 (2011) 279-283 
%
% [2] Tseitlin et al. Corrections for sinusoidal background and
% non-orthogonality of signal channels in sinusoidal magnetic field scans.
% J. Mag. Res. 223 (2012) 80-84

%% Constants

gamma=1.7608e7;     % Gyromagnetic Ratio, [rad/s*G]
g2f=gamma/(2*pi);   % 2.8024e6; Field to Frequency

%% Input parameters
% Here the parameters from the par structure are converted to standalone
% variables.

cf=Par.cf;      % Center Field [G]
Hm=Par.sw;      % Peak-to-peakp modulation amplitude [G]
Vm=Par.sf;      % Modulation frequency    [Hz]
rs=Spec;        % Rs signal
fwhm=Par.fwhm;  % Filter

if isfield(Par,'debug')
%     plot(Time,real(rs),Time,imag(rs));
%     title('Raw Data');
%     pause;
%     clf;
end

if isfield(Par,'nH')==1
    nH=Par.nH; % Number of Harmonics
else
    nH=1; % Defults to 1 if omitted
end

%% Derived parameters
% Here parameters calculated from the inputs are determined.

% Amplitude and Orthogonality Correction

if isfield(Par,'ACorr')==1
    ACorr=Par.ACorr;
else
    ACorr=1;
end

if isfield(Par,'Ortho')==1
    Ortho=Par.Ortho;
else
    Ortho=0;
end

Ortho_exp=exp(1i*Ortho/180*pi);
rs=ACorr*real(rs)+1i*imag(rs*Ortho_exp);

M=length(rs);
ss=size(rs);

if ss(1)>ss(2)
    rs=transpose(rs);   % Checks and corrects the data for proper dimensions
end

tb=(Time(2)-Time(1))*1e-9;  % Time base (sampling period) for signal stored in array [s]
t=(0:(M-1))*tb;             % Time vector (raw data)
Vmax=g2f*Hm;                % Max possible RS signal frequency
Ns=2*ceil(Vmax/Vm);         % Min number of points in the frequency domain(=the time domain)
P=1/Vm;                     % Scan period
ts=(0:(Ns-1))*(P/Ns);       % time vector for final filtering & interpolation
Wm=2*pi*Vm;                 % Angular scan frequency
Nc=round(1/(tb*Vm));        % Points per period
Nfc=floor(M/Nc);            % Number of full cycles

%% Error & Warning check

% Checks for presence of at least one full RS cycle
if Nfc==0
    disp('ERROR: less than a full cycle');
    error('RS signal must have at least one full cycle');
end

%% Interleaving data to one periodic cycle
% This function interleaves the full cycles into a single averaged cycle.
% This is done in the forrier domain.

rs_i=InterleavingCycles(t,ts,rs,Vm,Nfc,Ns,Wm,'fast');

if isfield(Par,'debug')
    rsOut.OneCyc=rs_i;
%     plot(real(rs_i)); hold on;
%     plot(imag(rs_i));
%     title('Averaged Full Cycle');
%     pause;
%     clf;
end

%% Position of 1st point and Phase correction
% Here the user determined first point, phase, and scan direction are
% applied to the data. If unspecified, a GUI pops up to determine.

if isfield(Par,'fp')==0 || isfield(Par,'ph')==0 || isfield(Par,'up')==0
    [fp,ph,up]=findFPnPH(Time,Spec,Par,'Sin',Ns);
else
    fp=Par.fp;      % First point of the cycle to be deconvolved
    ph=Par.ph;      % phase correction [degrees]
    up=Par.up;      % Scan Direction
end

rs_ii=rs_i*exp(1i*ph/180*pi);   % Phase correction

if isfield(Par,'debug')
    rsOut.PhCor=rs_ii;
%     plot(real(rs_ii)); hold on;
%     plot(imag(rs_ii));
%     title('Phase Corrected');
%     pause;
%     clf;
end

rs_iii=circshift(rs_ii',fp)';   % Circular shift to find 1st point

if isfield(Par,'debug')
    rsOut.fpCor=rs_iii;
%     plot(real(rs_iii)); hold on;
%     plot(imag(rs_iii));
%     title('First Point Corrected');
%     pause;
%     clf;
end

%% Driving function
% Here the driving function is calculated. This is based on equation (1) in
% reference [1].

rs=rs_iii;                  % Data are renamed
t=ts;                       % Time vector is renamed
tb=ts(2);                   % new time base
if strcmp(up,'up')==1
    WF=-cos((2*pi*Vm*t));      % Waveform
else
    WF=cos(2*pi*Vm*t);      % Waveform
end
W=gamma*Hm/2*WF;            % waverform with proper y axis in rad/s
dr=exp(1i*cumsum(W)*tb);   % driving function for 1 cycle

%% Separation Up from Down
% Here the data are taken into the forrier domain to seperate the halves of
% the cycles and to isolate the background. These calculations are based on
% reference [2].

[v RS]=fftM(t,rs);

if strcmp(up,'up')==1
    in=v<-(Vm*nH);
    a=ifft(ifftshift(RS.*in)); % up scan
    jn=v>(Vm*nH);
    b=ifft(ifftshift(RS.*jn)); % down scan
    aa=a(1:Ns/2);
    bb=b(Ns/2+1:Ns);
    bga=a(Ns/2+1:Ns);
    bgb=b(1:Ns/2);
else
    in=v>(Vm*nH);
    a=ifft(ifftshift(RS.*in)); % up scan
    jn=v<-(Vm*nH);
    b=ifft(ifftshift(RS.*jn)); % down scan
    aa=a(1:Ns/2);
    bb=b(Ns/2+1:Ns);
    bga=a(Ns/2+1:Ns);
    bgb=b(1:Ns/2);
end

if isfield(Par,'debug')
    rsOut.uSep=aa;
    rsOut.dSep=bb;
    rsOut.bguSep=bga;
    rsOut.bgdSep=bgb;
%     plot(real([aa bb])); hold on;
%     plot(real([bgb bga]));
%     title('Real, First Approximate BG');
%     pause;
%     clf;
end

t1=t(1:Ns/2);
t2=t(Ns/2+1:Ns);

%% BG removal
% Here the isolated backgrounds from the previous section are removed from
% the data. The H_amplit function is employed in this section. This, again,
% is detailed in reference [2]. 

if nH>0 && nH<=2
    % Up Scan
    x=2*pi*Vm*t1;
    for q=1:nH
        cosCmp(q,:)=cos(q*x); % Array of cosine components
        sinCmp(q,:)=sin(q*x); % Array of sine components
    end
    
    for kk=1:2 % real and imaginary
        if kk==1
            [r,cons]=H_amplit(t2,real(bga),Vm,nH);
            bg=0;
            for q=1:nH
                bg=bg+r(1,q)*sinCmp(q,:)+r(2,q)*cosCmp(q,:);
            end
            bg=bg+cons;
        else
            [r,cons]=H_amplit(t2,imag(bga),Vm,nH);
            tmp=0;
            for q=1:nH
                tmp=tmp+r(1,q)*sinCmp(q,:)+r(2,q)*cosCmp(q,:);
            end
            tmp=tmp+cons;
            bg=bg+1i*tmp;
        end
    end
    
    % Down Scan
    x=2*pi*Vm*t2;
    for q=1:nH
        cosCmp(q,:)=cos(q*x); % Array of cosine components
        sinCmp(q,:)=sin(q*x); % Array of sine components
    end
    
    for kk=1:2 % real and imaginary
        if kk==1
            [r,cons]=H_amplit(t1,real(bgb),Vm,nH);
            bg1=0;
            for q=1:nH
                bg1=bg1+r(1,q)*sinCmp(q,:)+r(2,q)*cosCmp(q,:);
            end
            bg1=bg1+cons;
        else
            [r,cons]=H_amplit(t1,imag(bgb),Vm,nH);
            tmp=0;
            for q=1:nH
                tmp=tmp+r(1,q)*sinCmp(q,:)+r(2,q)*cosCmp(q,:);
            end
            tmp=tmp+cons;
            bg1=bg1+1i*tmp;
        end
    end
    
    aaa=aa-bg;
    bbb=bb-bg1;
else
    aaa=aa;
    bbb=bb;
end

if isfield(Par,'debug')
    rsOut.uCor=aaa;
    rsOut.dCor=bbb;
end

%% Deco for Up Scan
% Here the driving function is removed from the up scan.

drA=dr(1:Ns/2);     % Up scan portion of the driving function
aaaa=aaa.*drA;
[~,A]=fftM(t,aaaa);
[~,D]=fftM(t,drA);
A=A./(D);

%% Deco for Down Scan
% Here the driving function is removed from the down scan.

drB=dr(Ns/2+1:Ns);  % Down scan portion of the driving function
bbbb=bbb.*drB;
[~,B]=fftM(t,bbbb);
[v,D1]=fftM(t,drB);
B=B./(D1);

%%

h=(v*2*pi)/gamma;
in=abs(h)<Hm/2;
A=A(in);
B=B(in);
h=h(in);

%% Post-filtering
% Here a gaussian filter is applied to the data. This employs the
% mygaussian function.

filter=mygaussian(h,0.05*fwhm*10);
sm=sum(filter);
if sm>0
    filter=filter/sum(filter);
    A=conv(A,filter,'same');
    B=conv(B,filter,'same');
else
    disp('Gaussian post-filter is ignored')
end

A=zeroLine(A,0.05);
B=zeroLine(B,0.05);
A=real(A);%+imag(hilbert(imag(A)));
B=real(B);%+imag(hilbert(imag(B)));
H=h+cf;

if isfield(Par,'debug')
    assignin('base','rsOut',rsOut);
end
end

function rs_x=InterleavingCycles(t,ts,rs,scan_freq,Nfc,Ns,Wm,method)
%% Forrier domain rapid scan cycle averaging
% This program is used within Sin to convert the input data to the
% forrier domaine and then average the signals into a sigle cycle

P=1/scan_freq;      % period
tb=t(2);            % Time base, asumes the vector begins at 0 ns
in=t<Nfc*P;
K=(-Ns/2):(Ns/2-1);
rs_i=rs(in);        % rs with Bfc full cyles
M=length(rs_i);
t_i=t(in);          % Time vector

switch(method)
    case 'slow'     % Currently not working, extreemly slow.
        RS_i=zeros(1,Ns);
        
        for k=1:length(K)
            tmp=exp(-1i*K(k)*Wm*t_i);
            RS_i(k)=rs_i*tmp';
        end
        
        rs_x=fft(fftshift(RS_i))*tb/ts(2)/Nfc/Ns;
        disp('method 1');
        
    case 'fast'     % Working
        [v RS]=fftM(t_i,[rs_i rs_i]);
        RS_i=interp1(v,RS,K/P);
        RS_i(isnan(RS_i))=0;
        rs_x=ifft(fftshift(RS_i))/M*Ns/2;
        
    otherwise
        disp('Select appropriate case');
end

end

function y=mygaussian(h,FWHM)
%% Gaussian Filter
% This subfunction is used in Sin to apply a gaussian filter

% Normalized to 1;
Hpp=FWHM/sqrt(2*log(2));
A=sqrt(2/pi)/Hpp;
x=2*(h/Hpp).^2;
y=A*exp(-x);

end

function [r,cons]=H_amplit(xi,yi,Fm,nH)
%% Sinusoidal BG Removal
% This function is used in Sin to remove the RS background

t=2*pi*Fm*xi;
f=yi;
fs=(f+fliplr(f))/2;
fa=(f-fliplr(f))/2;
for q=1:nH
    s(q,:)=sin(q*t);
    c(q,:)=cos(q*t);
end
o=c(1,:)*0+1;

%% Symmetric part

for q=1:nH
    if rem(q,2)==1
        vs(q,:)=s(q,:);
    elseif rem(q,2)==0
        vs(q,:)=c(q,:);
    end
end
vs(nH+1,:)=o;
f=vs*fs';
T=vs*vs.';
xs=T\f;

%% Asymmetric part

for q=1:nH
    if rem(q,2)==1
        va(q,:)=c(q,:);
    elseif rem(q,2)==0
        va(q,:)=s(q,:);
    end
end
f=va*fa';
T=va*va.';
xa=T\f;

%% Output

for q=1:nH
    if rem(q,2)==1
        r(1,q)=xs(q);
        r(2,q)=xa(q);
    elseif rem(q,2)==0
        r(1,q)=xa(q);
        r(2,q)=xs(q);
    end
end
cons=xs(end);

end

function [res,tmp]=zeroLine(spectrum,extent)
%% Spectral Offset Removal
% Puts the begining of the spectrum at 0

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

function [fpok,phok,upok]=findFPnPH(Time,Spec,Par,Type,NPC)
%% GUI for first point and Phaase determination

%% Initialization

fpok=[];
phok=[];
upok=[];

if isfield(Par,'fp')==0
    fp=1;
    Par.fp=fp;
else
    fp=Par.fp;
end
 
if isfield(Par,'ph')==0
    ph=0;
    Par.ph=ph;
else
    ph=Par.ph;
end

if isfield(Par,'up')==0
    up='up';
    Par.up=up;
else
    up=Par.up;
end

%% Main window

hwin=figure('Visible','off','Name','First Point and Phase GUI');
bgclr=get(hwin,'Color');

%% Axes

axes('Units','pixels');
set(hwin,'position',[0,0,1000,600]);

%% Buttons OK & Cancel

uicontrol(hwin,'Style','pushbutton','String','OK','Position',[10,10,80,25],'Callback',{@OKFcn},'BackgroundColor',bgclr); 
uicontrol(hwin,'Style','pushbutton','String','Cancel','Position',[100,10,80,25],'Callback',{@CancelFcn},'BackgroundColor',bgclr);

%% First Point Slider

hfp=uicontrol(hwin,'Style','slider','Position',[10,50,980,25],'Min',-NPC,'Max',NPC,'Value',fp,'SliderStep',[1/(NPC) (1/NPC)*10],'Callback',{@fpFcn});
hfplbl=uicontrol(hwin,'Style','text','String',['First Point: ' num2str(fp)],'Position',[10,75,100,20],'HorizontalAlignment','center','BackgroundColor',bgclr);

%% Phase Slider

hph=uicontrol(hwin,'Style','slider','Position',[10,105,980,25],'Min',0,'Max',360,'Value',ph,'SliderStep',[1/36000 1/360],'Callback',{@phFcn});
hphlbl=uicontrol(hwin,'Style','text','String',['Phase: ' num2str(ph)],'Position',[10,130,100,20],'HorizontalAlignment','center','BackgroundColor',bgclr);

%% Up/Down Switch

hup=uibuttongroup('Visible','on','Title','Scan Direction','Position',[0.92 0.25 0.08 0.13],'SelectionChangedFcn',@upFcn);
uicontrol(hup,'Style','radiobutton','String','up','Position',[10 30 50 30],'HandleVisibility','off');
uicontrol(hup,'Style','radiobutton','String','down','Position',[10 0 50 30],'HandleVisibility','off');

%% Move the GUI to the center of the screen

movegui(hwin,'center');

%% Plot a first estimation

[fp,ph,up]=compute(Time,Spec,Par);

%% Make the GUI visible

set(hwin,'Visible','on');

%% Callback functions

    function CancelFcn(~,~)
        % Just close the window
        fpok=1;
        phok=0;
        upok='up';
        uiresume(gcbf);
        close(hwin);
        return
    end
  
    function OKFcn(~,~)
        % Return the current estimation and close the window
        fpok=fp;
        phok=ph;
        upok=up;
        uiresume(gcbf);
        close(hwin);
    end

    function fpFcn(~,~)
        % Change First Point
        Par.fp=round(get(hfp,'Value'));                
        [fp,~,~]=compute(Time,Spec,Par);
        set(hfplbl,'String',['First Point: ' num2str(fp)]);
    end

    function phFcn(~,~)
        % Change Phase
        Par.ph=get(hph,'Value');                
        [~,ph,~]=compute(Time,Spec,Par);
        set(hphlbl,'String',['Phase: ' num2str(ph)]);
    end

    function upFcn(~,~)
        % Change Up/Down
        Par.up=get(hup.SelectedObject,'String');
        [~,~,up]=compute(Time,Spec,Par);
    end

    function [fp,ph,up]=compute(Time,Spec,Par)
        % Compute and plot
        switch Type
            case 'Lin'
                [Field,ru,~,rd,~]=Lin(Time,Spec,Par);
                plot(Field,ru,Field,fliplr(rd));
                set(gca,'OuterPosition',[0 150 970 450]);
                axis tight;
                fp=Par.fp;
                ph=Par.ph;
                up=Par.up;
                
            case 'Sin'
                [Field,Up,Down]=Sin(Time,Spec,Par);
                plot(Field,Up,Field,Down); 
                set(gca,'OuterPosition',[0 150 970 450]);
                axis tight;
                fp=Par.fp;
                ph=Par.ph;
                up=Par.up;
        end        
    end

uiwait(gcf);

end