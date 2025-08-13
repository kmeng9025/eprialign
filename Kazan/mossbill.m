function [x, y] = mossbill(Sys, Exp, Opt)
A(1) = safeget(Sys, 'isoshift', 0); %SHIFT=XC(J,1) 
A(2) = safeget(Sys, 'lw', 0.1); %GAMMA=XC(J,2) + wift(J,is)
A(3) = safeget(Sys, 'K', 0); % DELTAEQ=XC(J,3) + qift(J,is) (mm/s)
A(4) = safeget(Sys, 'etha', 0); %      ETA=XC(J,4)
A(5) = safeget(Sys, 'D', 0); % D=XC(J,5)
A(6) = safeget(Sys, 'ED', 0); %  EVERD=XC(J,6)
A(7:9) = safeget(Sys, 'Qpa', [0, 0, 0]); % euler angles for Quad
%       AL=XC(J,7)
%       BE=XC(J,8)
%       GA=XC(J,9) % 
A(10:12) = safeget(Sys, 'g', [2, 2, 2]); % g-value 
% GX=XC(J,10)
% GY=XC(J,11)
% GZ=XC(J,12)
A(13:15) = safeget(Sys, 'A', [0, 0, 0]); % HF (kG)
% AXX=XC(J,13)
% biso = axx
% AYY=XC(J,14)
% AZZ=XC(J,15)
A(16:18) = safeget(Sys, 'Apa', [0, 0, 0]);
% AL1=XC(J,16)
% BE1=XC(J,17)
% GA1=XC(J,18)
A(19) = safeget(Sys, 'D', 0); % DP=XC(J,19)
A(20) = safeget(Sys, 'ED', 0); % EVERDP=XC(J,20)
A(21:23) = safeget(Sys, 'J', [0, 0, 0]); %
% GPX=XC(J,21) % looks like exchange
% GPY=XC(J,22)
% GPZ=XC(J,23)
A(24:27) = 0; 
% AJ= -2. * XC(J,24)
% XJ= -2. * XC(J,25)
% YJ= -2. * XC(J,26)
% ZJ= -2. * XC(J,27)
npoints = safeget(Exp, 'nPoints', 100);
Range = safeget(Exp, 'Range',[-10, 10]);
dRange = Range(2)-Range(1);
%%%%%%%%%%%% Experimental parameters
B(1) = floor(npoints/2); %CENTER = P(1)
B(2) = dRange/npoints; %STEP = P(2)
B(3) = safeget(Exp, 'Field', 100); %HEXT = P(3)
B(4) = safeget(Exp, 'T', 4); %TEMP = P(4)
B(5) = safeget(Exp, 'rfactor', 1); %rfaktor = P(5)
B(6) = safeget(Exp, 'trift', 0); %trift(1,1) = P(6) 
B(7) = safeget(Exp, 'qift', 0); %qift(1,1) = P(7)
B(8) = safeget(Exp, 'wift', 0); %wift(1,1) = P(8)
B(9) = safeget(Exp, 'isoflg', 0); %iso_flg(1) = P(9)
B(10) = safeget(Exp, 'iSym', 0); %ISYM = P(10)
B(11) = safeget(Opt, 'nKnots', 10); %NPLONG = P(11) % number of points for theta
B(12) = safeget(Exp, 'iPol', 0); %IPOL(1) = P(12)
% IPOL=-3   INTENSITY TENSOR AND INTENSITY VECTOR ARE TO BE            
%          CALCULATED. NO INPUT FOR EX,EY,EZ NECESSARY                
% IPOL=-2   POWDER MEAN VALUE; NO INPUT FOR EX,EY,EZ NECESSARY         
% IPOL=-1   PERPENDICULAR CONFIGURATION (SPLIT COIL)                   
%          EX,EY,EZ IS TO BE PARALLEL TO THE EXTERNAL FIELD           
% IPOL=0    UNPOLARIZED GAMMA-RAY PARALLEL TO EX,EY,EZ                 
% IPOL=1    CIRCULAR POLARISATION,WAVE VECTOR PARALLEL EX,EY,EZ        
% IPOL=2    LINEAR POLARISATION; IN 57FE (EX,EY,EZ) IS THE VECTOR      
%          PERPENDICULAR TO THE WAVE VECTOR AND PERPENDICULAR TO      
%          THE VECTOR POTENTIAL A
B(13) = 0; %ISUB = P(13) number of subspectra
B(14) = 2; %NSTAT(1) = P(14) % % 2s+1 (ground)
B(15) = 4; %NPSTAT(1) = P(15) % 2s+1 (excited)
B(16) = 1; %IRELAX(1,1) = P(16)
B(17) = safeget(Opt, 'Verbosity', 0); % verbosity
% B(17) = 1; %ROI(1, 1) = P(17)
B(18) = npoints; %ROI(1, 2) = P(18)

try
    y = CalcMoss_Bill(A', B');
catch
     error('check CalcMoss_Bill.dll')
end
x = [0:(length(y)-1)]-B(1); %points
x = x*B(2);
y = y(:) - y(1);
x = x(:);
% coll = abs(rand);
% figure(22); plot(x(1:end), y(1:end)-max(y), 'Color', [1-rand, 0, coll]); hold on;
% title('Mossbauer spectrum')
% xlabel('Velosity, mm/s')