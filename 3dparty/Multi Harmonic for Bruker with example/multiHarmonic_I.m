function spI = multiHarmonic_I(x,hm,a,filter_width)
%% Outputs Absorption Line (not derivative)

% filter_width -in [G] units, Gaussian filter
% h - magnetic field vector
% Hpp - modulation amplitude [G]
% Yc - 2D array with spectra
% cutoff must be of the order of modulation amplitude; reduces overall noise
% filter _width is of the order and smaller than the undistorted linewidth; reduces reconstruction artifacts

ss=size(a);
if ss(1)<ss(2)
    Nh=ss(1);
    a=transpose(a);
else
    Nh=ss(2);
    
end
%%
[w S]=fftM(x,ifftshift(a));
S=transpose(S);
H=HnFilters(hm,w,Nh);

sigmaN=1;
alfa=conj(H)./sigmaN.^2;
SS=sum(alfa.*S);
HH=sum(alfa.*H);

%%  filter function
%w_cut=2*pi/cutoff;
%filter1=(abs(w)<w_cut)+0.0;
g=mygaussian(x-mean(x),filter_width);
%filter=conv(filter1,g,'same');

%%
Res=SS./(HH+max(abs(HH)*1e-6));
sp=real(fftshift((ifft(fftshift(Res)))));

spI=cumsum(zeroLine(sp,0.05));
if sum(g)~=0
    spI=conv(spI,g,'same');
end
spI=zeroLine(spI,0.05);






