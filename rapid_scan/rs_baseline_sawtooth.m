function [data, BG]=rs_baseline_sawtooth(B, data, dB, nH)

% data in the format (B, [-dB, +dB]) are assumed
np = size(data, 1);
nprj2 = size(data, 2);
shiftsize = round(np*dB/2/(max(B) - min(B)));

% idx1 = 1:nprj2/2;          % indexes for dn field projections
% idx2 = nprj2/2+1:nprj2;    % indexes for up field projections

idx1 = 1:2:nprj2;          % indexes for dn field projections
idx2 = 2:2:nprj2;    % indexes for up field projections

% Elimination of signal
data = real(data);
Rs=data+flipud(data);                % step1; elimination of asymmetrical part

R1=Rs;
R1(np/2+1:end,idx1)=Rs(np/2+1:end,idx2); %     step2; rearranging 
R1(np/2+1:end,idx2)=Rs(np/2+1:end,idx1); %
a0=R1(:,idx2);
b0=R1(:,idx1);
a=circshift(a0,-shiftsize);       % step 3; circularly shifting 
b=circshift(b0,shiftsize);
ab=a-b;                          % step4; subtraction

t=(0:np-1)/np;
BG=zeros(np,nprj2/2);
dBG=zeros(np,nprj2/2);
tau_f = 2*shiftsize/np;

if tau_f*nH > 1/5
  disp('Too large harmonics !!!')
end

for k=1:nH                        % Calculation of BG harmonics amplitudes
    sn=sin(2*pi*k*t);
    cs=cos(2*pi*k*t);
    SN=repmat(sn',1,nprj2/2);
    CS=repmat(cs',1,nprj2/2);
    ss=sum(ab.*SN)';
    Den=-1/sin(pi*tau_f*k)/2;
    cc=2*ss/np;
    CC=repmat(cc',np,1);
    BG=BG+CC.*CS*Den;
    dBG=dBG+CC.*SN;
end

isshow = true;
if isshow
  slice = 1; sl1 = idx1(slice); sl2 = idx2(slice);
  figure(100); clf;
  subplot(4,1,1); plot(B, real(data(:,[sl1,sl2]))); axis tight
  subplot(4,1,2); plot(B, real(a(:,slice)), B, real(b(:,slice))); axis tight
  
  subplot(4,1,3); plot(B, real(ab(:,slice)), B, real(dBG(:,slice)), 'r'); axis tight
  subplot(4,1,4); plot(B, real(a(:,slice)+b(:,slice))/2, ...
    B, cumsum(ab(:,slice))*dB/2, 'm', B, BG(:,slice), 'r'); axis tight
%   subplot(3,1,3); plot(B, real(a+b)-2*BG); axis tight
end

data = (a+b)/2 - BG;
return;