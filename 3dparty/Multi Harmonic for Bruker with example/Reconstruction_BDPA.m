clear all; clc;
%% read data from file
file_name='19_spu_BDPA_040313.dsc';
[h,Y,N,Hpp,Fm] = readBES3Tm(file_name);   
% h - magnetic field vector [G]
% N - total number of spectra  in & out of phase
% Hpp - modulation amplitude [G]
% Fm - modulation frequency 
%% manual phase correction
% we take only real part; in ideal case imaginary part = zero.
ph=1.6; % adjustable phase in degrees
Yc=Y(1:2: end,:)+1i*Y(2:2: end,:); % quadrature complex signal   
   for k=1:4
        subplot(2,2,k);
        set(gca,'FontSize',14)
        sp=Yc(k,:)*exp(1i*k*ph/180*pi);
        plot(h,real(sp),h,imag(sp),'linewidth',3);      
        tx=['Number of harmonic:' num2str(k)];  title(tx);        
        axis tight
   end
 Yc=real(Yc);  % we remove imaginary part
 
 %% Reconstruction 
cutoff=2*Hpp; % [G]
filter_width=0.01; % [G] 
z = multiHarmonic(h,Hpp,Yc,cutoff,filter_width); % recovered  1st derivative

filter_width=0.004; % [G] 
zI = multiHarmonic_I(h,Hpp,Yc,filter_width);

% filter_width -in [G] units, Gaussian filter 
% h - magnetic field vector
% Hpp - modulation amplitude [G]
% Yc - 2D array with spectra
% cutoff must be of the order of modulation amplitude; reduces overall noise 
% filter _width is of the order and smaller than the undistorted linewidth; reduces reconstruction artifacts 

%% Show results

subplot(2,1,1);
set(gca,'FontSize',16);

y1=Y(1,:); % raw data 1st harmonic signal
plot(h,y1/max(y1),h,z/max(z),'linewidth',3); 
axis tight; 

legend('1^s^t harmonic','1^s^t derivative');
xlabel 'Magnetic Field' 

subplot(2,1,2);
set(gca,'FontSize',16);

y1=Y(1,:); % raw data 1st harmonic signal
y1I=cumsum(zeroLine(y1,.05)); % raw data 1st harmonic signal
plot(h,y1I/max(y1I),h,zI/max(zI),'linewidth',3); 
axis tight; 

legend('Integral of 1^s^t harmonic','Absorption');
xlabel 'Magnetic Field' 
