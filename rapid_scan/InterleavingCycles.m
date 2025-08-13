function rs_x=InterleavingCycles(t,ts,rs,scan_freq,Nfc,Ns,Wm,method)
P=1/scan_freq; % period
tb=t(2);
inx=t>(t(end)-P+tb);
rs_ext=[rs rs(inx)];
in=t<Nfc*P;  %
K=(-Ns/2):(Ns/2-1);

rs_i=rs(in); % rs with Bfc full cyles
M=length(rs_i);
t_i=t(in);   % time vector

switch(method)
    case 'slow'
        %%               
        RS_i=zeros(1,Ns);      
        for k=1:length(K);
            tmp=exp(-1i*K(k)*Wm*t_i);
            RS_i(k)=rs_i*tmp';
        end
        rs_x=fft(fftshift(RS_i))*tb/ts(2)/Nfc/Ns;        
        
        %%
    case 'fast'
        %%
        [v RS]=fftM(t_i,[rs_i rs_i]);
        RS_i=interp1(v,RS,K/P);
        RS_i(isnan(RS_i))=0;   %% 
        rs_x=ifft(fftshift(RS_i))/M*Ns/2;
        %plot(v,abs(RS_y),'x')
        %%
    case '3'
        
    otherwise
        disp('Select appropriate case');
        
end
