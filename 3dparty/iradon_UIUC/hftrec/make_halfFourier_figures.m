%% Load data

% For these simulations, readouts will be in the horizontal direction, and
%  phase encoding will be in the vertical direction

load MID242_phantom I; %turbo spin echo image
SE=I;

load MID244_phantom I; %turbo flash image
GE=I;

%% Fully sampled images
imwrite(abs(SE),'spin_echo_full_Mag.png')
imwrite(mod((angle(SE)+3*pi/2)/(2*pi),1),'spin_echo_full_Ph.png')

imwrite(abs(GE),'grad_echo_full_Mag.png')
imwrite(mod((angle(GE)+3*pi/2)/(2*pi),1),'grad_echo_full_Ph.png')

%% half-Fourier encoding with zeropadded Fourier reconstruction
% SE_half = fftshift(fft2(SE));
% SE_half(1:128,:) = 0;
% SE_half = ifft2(ifftshift(SE_half));
% 
% imwrite(abs(SE_half),'spin_echo_half_Mag.png')
% % imwrite(mod((angle(SE_half)+3*pi/2)/(2*pi),1),'spin_echo_half_Ph.png')
% 
% GE_half = fftshift(fft2(GE));
% GE_half(1:128,:) = 0;
% GE_half = ifft2(ifftshift(GE_half));
% 
% imwrite(abs(GE_half),'grad_echo_half_Mag.png')
% % imwrite(mod((angle(GE_half)+3*pi/2)/(2*pi),1),'grad_echo_half_Ph.png')

%% partial-Fourier encoding with advanced reconstruction

SE_fft = fftshift(fft(SE));
GE_fft = fftshift(fft(GE));

for sym=[0 8 16 32] %size of symmetric portion (0 is half-fourier)
    it = 0;
    
    SE_temp = SE_fft;
    SE_temp(1:(129-sym/2-1),:)=0;
    imwrite(abs(ifft(ifftshift(SE_temp))),['spin_echo_partial' num2str(sym) '_it' num2str(it) '_Mag.png'])
%     imwrite(mod((angle(ifft(ifftshift(SE_temp)))+3*pi/2)/(2*pi),1),['spin_echo_partial' num2str(sym) '_it' num2str(it) '_Ph.png'])

    GE_temp = GE_fft;
    GE_temp(1:(129-sym/2-1),:)=0;
    imwrite(abs(ifft(ifftshift(GE_temp))),['grad_echo_partial' num2str(sym) '_it' num2str(it) '_Mag.png'])
%     imwrite(mod((angle(ifft(ifftshift(GE_temp)))+3*pi/2)/(2*pi),1),['grad_echo_partial' num2str(sym) '_it' num2str(it) '_Ph.png'])
  
  for it=1:4
    SE_part = hftrec(SE_fft((129-sym/2):end,:),256,sym/2+1,it,0);
    imwrite(abs(ifft(ifftshift(SE_part))),['spin_echo_partial' num2str(sym) '_it' num2str(it) '_Mag.png'])
%     imwrite(mod((angle(ifft(ifftshift(SE_part)))+3*pi/2)/(2*pi),1),['spin_echo_partial' num2str(sym) '_it' num2str(it) '_Ph.png'])
    
    GE_part = hftrec(GE_fft((129-sym/2):end,:),256,sym/2+1,it,0);
    imwrite(abs(ifft(ifftshift(GE_part))),['grad_echo_partial' num2str(sym) '_it' num2str(it) '_Mag.png'])
%     imwrite(mod((angle(ifft(ifftshift(GE_part)))+3*pi/2)/(2*pi),1),['grad_echo_partial' num2str(sym) '_it' num2str(it) '_Ph.png'])
    
  end
end

