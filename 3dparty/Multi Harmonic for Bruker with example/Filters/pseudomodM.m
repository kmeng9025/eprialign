function sp=pseudomodM(x,y,hm,zf)
% x-  magnetic field;
% y - absorption aspectrum hm - modulation amplitude (peak-to-peak)
% zf-  Number of the filter zero after which filter =0.
%%
zf1=3.8317;   %  1st zero
zf2=7.0156;   %  2nd zero
zf3=10.1735;  %  3rd zero
zf4=13.3237;  % 4th zero

%%

sum(zf==[1 2 3 4]);
if sum(zf==[1 2 3 4])==0
    disp(' zf must be 1,2,3 or 4')
    sp=0*y;
else
    [v,S0]=fftM(x,y);
    z=hm*pi*v;
    J1=1i*besselj(1,z);
    if zf==1; in=abs(z)>zf1; end;
    if zf==2; in=abs(z)>zf2; end;
    if zf==3; in=abs(z)>zf3; end;
    if zf==4; in=abs(z)>zf4; end;
    J1(in)=0;
    S1=S0.*J1;
    sp=ifft(ifftshift(S1));
    %pltc(J1);
end
sp=4*sp/hm;
%%

