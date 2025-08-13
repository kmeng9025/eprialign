% Matlab current dirrectory should be to one with these files
% mex -g.... means enable debug
% to do debug:
%  1. use command "dfdev MrWinML.dll" to launch debug
%       with Visual Fortran. (run in "totalcommander" or from "Start->run")
%  2. In the CVF program in the option
%       "Project->Settings...->Debug->Executable..." put the matlab.exe 
%       including the full path e.g. "D:\MatLab6p5\bin\win\matlab.exe"
%  3. Add to the workspace all neccessary fortran source files
%       "Project->Add to Project->Files"
%  4. Insert breakpoints.
%  4. To start debuging press "Go"
%       A massage will appear telling that matlab.exe does not contain
%       debuging information. It is normal. Just press OK. 
%  5. New Matlab will start. Start the program, proceed with debugging

% to compile, execute following strings:
% 1D data
mex -g MrWinML.f cummod.for dataman2.for fftpack.F luxlib2.for macman2.for mr2.for mrlib1.for procmod.for simepr1.for simese1.for userin.F
% or 2D ESEEM vs Bo
mex -g MrWinML2.f cummod.for dataman2.for fftpack.F luxlib2.for macman2.for mr2.for mrlib1.for procmod.for simepr1.for simese1.for userin.F
%
% IMPORTANT: for the final compilation remove "-g" (debug). Otherwise it
% will not work on any other computers, which have no fortran compiler
% installed
%%%%%%%%%% test simulation %%%%%%%%%%%%%%%
% 1D data
%g-value
Sys(1:3) = [2.0, 2.0, 2.3];% g-values
Sys(4:6) = 0; % angles
Sys(7) = 10; % linewidth
Sys(8) = 1;                 % number of nuclei (10 maximum)
Sys(9:11) = [0.666, 0.666, 0.666];  % HF coupling [MHz]
Sys(12:14) = 0;             % HF tensor rotation [degree]
Sys(15:17) = [0.3, 0.1, -0.4];    % quadrupole coupling
Sys(18:20) = 0;             % Q tensor rotation [degree]
Sys(21) = 0.4037;           % gn - nuclear g-value
Sys(22) = 3;                % 2*I+1

Exp(1) = 800;               % nPoints
Exp(2) = 10000;             % MW freq. MHz
Exp(3) = planck*Exp(2)/2.3/bmagn*1e10;              % Magnetic field [G]
Exp(4) = 0.2;                 % tau [us]
Exp(5) = 15;                % maximum freq. [MHz]
Exp(6) = 1;                 % xtype ?????

Opt(1) = 80;                % nKnots
Opt(2) = 0;                 % idebug
t = cputime;
y = MrWinML(Sys.', Exp.', Opt.');
s = cputime - t;
disp(['simulations time is: ', num2str(s), ' s.']);
y(1) = 0;
figure; 
plot(linspace(0, Exp(5), Exp(1)).', y);

%%%%%%%%%% test simulation %%%%%%%%%%%%%%%
% 2D data
%g-value
clear;
Sys(1:3) = [2.0, 2.0, 2.3];% g-values
Sys(4:6) = 0; % angles
Sys(7) = 10; % linewidth
Sys(8) = 2;                 % number of nuclei (10 maximum)
Sys(9:11) = [0.666, 0.366, -0.666];  % HF coupling [cm-4]
Sys(12:14) = 0;             % HF tensor rotation [degree]
Sys(15:17) = [0.3, 0.1, -0.4];    % quadrupole coupling [cm-4]
Sys(18:20) = 0;             % Q tensor rotation [degree]
Sys(21) = 0.4037;           % gn - nuclear g-value
Sys(22) = 3;                % 2*I+1

Sys(23:25) = [0.666, 0.366, -0.666];  % HF coupling [cm-4]
Sys(26:28) = 0;             % HF tensor rotation [degree]
Sys(29:31) = [0.3, 0.1, -0.4];    % quadrupole coupling [cm-4]
Sys(32:34) = 0;             % Q tensor rotation [degree]
Sys(35) = 0.4037;           % gn - nuclear g-value
Sys(36) = 3;                % 2*I+1

mwFreq = 10000;
Exp(1) = 800;               % nPoints
Exp(2) = 50;                % nFpoints
Exp(3) = planck*mwFreq/2.3/bmagn*1e10;              % Start Magnetic field [G]
Exp(4) = planck*mwFreq/2.0/bmagn*1e10;              % Last Magnetic field [G]
Exp(5) = mwFreq;             % MW freq. MHz
Exp(6) = 0.2;                 % tau [us]
Exp(7) = 10;                % maximum freq. [MHz]
Exp(8) = 1;                 % xtype [0]-none, [1]-gaussian, [2]-lorentzian

Opt(1) = 80;                % nKnots
Opt(2) = 0;                 % idebug
y = zeros(Exp(1), Exp(2));
t = cputime;
y = MrWinML2(Sys.', Exp.', Opt.');
s = cputime - t;
disp(['simulations time is: ', num2str(s), ' s.']);
y(1, :) = 0;
figure; 
%surf(linspace(Exp(3), Exp(4), Exp(2)),linspace(0, Exp(7), Exp(1)),  y);
h = imagesc(linspace(0, Exp(7), Exp(1)), linspace(Exp(3), Exp(4), Exp(2)), y.');
set(gca, 'YDir', 'normal');
xlabel('Frequency, MHz');
ylabel('Magnetic Field, G');