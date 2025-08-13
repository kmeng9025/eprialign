function I = iradonC(R,theta,N,domain)

ss=size(R); 
Np=ss(1);               % Number points in projection  
Na=ss(2);               % Number of projections  
Cnp = round(Np/2);       % center of the projection
% filtering in domain:   'freq' or 'time'

FilterLength=ceil(2*Np/128)*128;
filter=hamming(FilterLength);

if domain=='time';filtRamp=RampFilter(Np,domain);end
if domain=='freq';filtRamp=RampFilter(FilterLength,domain);end


Nmax=round(Np/sqrt(2));
if N>Nmax; N=Nmax; end;
I = zeros(N);           %Image

xLine = (1:N)-ceil(N/2);
x = repmat(xLine, N, 1);% X matrix    
y = rot90(x);           % Y matrix  
theta=theta*pi/180;
for i=1:Na
prj=R(:,i);
switch domain
    case 'time';
     prj=conv(prj,filtRamp');
     prj(length(prj):FilterLength)=0;% Zero pad projections
     prj=fftshift(fft(prj)).*filter; % hamming filter
     prj=real(ifft(ifftshift(prj)));
     prj=prj(round(Np-Np/2):round(Np+Np/2)); % filtered projection
    case 'freq';
     prj(length(prj):FilterLength)=0;% Zero pad projections   
     prj=fft(prj).*ifftshift(filter).*filtRamp; % hamming& ramp filters
     prj=real(ifft(prj));    
     prj=prj(1:Np); % filtered projection
end   




%Start Backprojecting
p=x.*cos(theta(i))+y.*sin(theta(i));
pLess=floor(p);

k=(pLess+Cnp);
inx=k<1;      k(inx)=1;
inx=k>(Np-1); k(inx)=(Np-1);
% in2=(pLess+Cnp+1); 

xxx=(p-pLess).*prj(k+1);
xxx=xxx+(pLess-p+1).*prj(k);
I=I+xxx;
end

I=I*pi/(2*Na);
