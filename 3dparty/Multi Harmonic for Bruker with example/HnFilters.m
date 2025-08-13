function H=HnFilters(hm,f,Nmax)
% hm - peak-to-peak modulation amplitude [G]
% f -  array of frequencies [Hz] 
% Nmax - Number of harmonics
L=length(f);
H=zeros(Nmax,L);

for n=1:Nmax
 w=2*pi*f;
 z=hm*w/2;
 JJ=besselj(n-1,z)+besselj(n+1,z);
 H(n,:)=hm/4/n*(1i)^(n-1)*JJ; 
end

%plot(abs(H'));

