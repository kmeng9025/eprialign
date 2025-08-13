%%
load CTPO15NPDT_FBPw0_60G_6dB_1.71d.mat

%%
radon_pars = [];
radon_pars.G = G;

recon_pars = [];
reconPars.L = 2.5;
recon_pars.nBins = [15,15,15,length(P{1})];
recon_pars.CenterXYZ = [-2,0,0];
recon_pars.MaxHarmonic = 150;

%%

A = iradon_d2d_4D_direct(B,P,radon_pars,recon_pars);
ibGUI(A)