function ss=fullPeriodRsDeco(sweep,freq,ph,rs)
% Full Period Triangle Deconvolution
% fullPeriodRsDeco(sweep,freq,rs);
% rs    =   rapid scan
% ss    =   slow scan
% freq  =   frequency in Hz
% sweep =   p-p amplitude in G
% ph - phase in degrees
L=length(rs);
h=fft(rs);
d = 1.4211e-008/sweep/freq; 
w=(1:L)*2*pi*freq;
A = exp(-1i*d*w.^2); % analytical function
L2=floor(L/2);
A1=A(1:L2);
A2=fliplr(conj(A1));
if 2*length(A1)<length(rs);
    B=[A1 A1(end) A2];
else
    B=[A1 A2];
end    

ss=ifft(h./B); % slow scan spectum
ss=ss*exp(1i*ph/180*pi); % fine phase correction


% t=(0:(L-1))/(L-1)*P*1e6;
% set(gca,'FontName','Times New Roman','FontSize',16)
% subplot(1,2,1);plot(t,real(ss)); axis tight; title 'M_y'
% xlabel 'Time, us'
% ylabel 'Intensity, a.u.'
% set(gca,'FontName','Times New Roman','FontSize',16)
% subplot(1,2,2);plot(t,imag(ss)); axis tight; title 'M_x'
% ylabel 'Intensity, a.u.'
% xlabel 'Time, us'
% saveas(gcf,'fig1_2','emf');