function fr=frank(N)
%N=16;
tw=0:(N-1);
Tw=repmat(tw,N,1);
fw=0:(N-1);
Fw=repmat(fw,N,1);
%Fw=Fw(:);
P=Fw.*Tw';
p=mod(P,N);
%subplot(2,1,1);
%imagesc(tw,fw,p);

%% Frak sequence
%subplot(2,1,2);
fr=exp(2*pi*1i*p(:)/N); % Frank sequence
%FR=fft(fr);
%xc=fftshift(ifft(FR.*conj(FR))); % autoCorrelation
%pltc(xc);
