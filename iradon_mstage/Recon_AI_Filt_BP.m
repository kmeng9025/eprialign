function mat_recFXD = Recon_AI_Filt_BP(mat_final, com)
%
% Last modified by Kang Ahn dd061208
% recon_gui version 0.1

imtype=com(1);
naz=com(2);
orgnbins = com(3); % before subsampling
nbins = size(mat_final,1); % after subsampling
nspec=com(4);
nplr=com(20);
% set nspec = 1 for spatial image
if imtype > 7
    nspec=1;
end
filtflag=com(13);
cutoff=com(15);
bpCodeFlag=com(18);
singlePrecision=com(19);
pinterpflag=com(21);
nAng_i=com(9:10);
ifactor=com(24);
if pinterpflag==0
    ifactor=1;
end

lshift=com(27);

if singlePrecision
    mat_final=single(mat_final); 
end

dataTypeStr=class(mat_final);

t=cputime;

% for backward compatibility of ifactor
if nAng_i(1)*nAng_i(2) < 1
    arg3=ifactor; 
else arg3=nAng_i; 
end




[mat_final, nplr, naz, nspec] = AngularInterpolation(mat_final, imtype, arg3, pinterpflag);
t=cputime-t;
fprintf('CPU time for Angular Interpolation: %5.3f\n',t);
% angular interpolation sinc/spline




t=cputime;
        mat_final_filt=FilterProjections(mat_final,imtype,filtflag,cutoff);    
t=cputime-t;
fprintf('CPU time for Filtering: %5.3f\n',t);

[nX, nY, nZ, nB]=get_image_dims(imtype,nbins);
mat_recFXD=zeros(nX*nY*nZ*nB,1,dataTypeStr);            

is1D=0; is2D=0;
switch imtype
    case {5, 6, 7, 11,12,13}
        is2D=1;
    case {8,9,10}
        is1D=1;
        mat_recFXD=mat_final;
end
clear mat_final;
%	mat_final_filt=reshape(mat_final_filt,nplr*naz*nspec*nbins,1);      
% disp('Reshape mat_final_filt for mex-file');

t=cputime;  
% sinc interpolation (pinterpflag=1) results in an interpolated
%	sinogram that starts with the initial angle = pi/2 - pi/N/2
% local interpolation (pinterpflag > 1) results in an interpolated
%	sinogram that starts with the initial angle = pi/2 - pi/Ni/2
% N = number of angles before interpolation
% Ni = number of angles after interpolation (Ni=N*ifactor)
% dd051118 KA
if pinterpflag>1
    iA(1)=pi/nplr/2;    % offset of initial polar angle
    iA(2)=pi/naz/2;     % offset of initial azimuthal angle
    iA(3)=pi/nspec/2;   % offset of initial spectral angle
else
    iA(1)=pi/nplr*ifactor/2;    % offset of initial polar angle
    iA(2)=pi/naz*ifactor/2;     % offset of initial azimuthal angle
    iA(3)=pi/nspec*ifactor/2;   % offset of initial spectral angle
end
bc=(nbins+1)/2+lshift/orgnbins*nbins;
%bc=(nbins+1)/2;

% if ((isunix | is2D) & bpCodeFlag==1) 
%     if isunix
%         fprintf('Warning -- FORTRAN back projection does not work in unix env.\n');
%         fprintf('   C code will be used.\n'); bpCodeFlag=3;
%     elseif is2D
%         fprintf('Warning -- FORTRAN back projection does not work for 2D.\n');
%         fprintf('   C code will be used.\n'); bpCodeFlag=3;
%     end
% end    
if (~singlePrecision) && bpCodeFlag==3
    bpCodeFlag=1;  %%%%%%%%%%%%%%  1 is for fortran  %%%%%%%%%%%%%%%%%55
    fprintf('Warning -- C back projection does not have double precision option.\n');
    fprintf('  FORTRAN code will be used.\n');bpCodeFlag=1;
end
if (imtype>1) && bpCodeFlag==3
    bpCodeFlag=2;            %%%%%%%%%%%%%%%%%%%%  2 is for MATLAB %%%%%%%%%%%%%%%
end

if is1D
    ; % do nothing
else
    if bpCodeFlag==1  %  FORTRAN
        if singlePrecision
            bc=single(bc);
            nplr=single(nplr);naz=single(naz);nspec=single(nspec);iA=single(iA);nbins=single(nbins);imtype=single(imtype);
            mat_recFXD=bp_xds_var_single(mat_final_filt,nplr,naz,nspec,iA(1),iA(2),iA(3),nbins,bc,imtype);
        else
            mat_recFXD=bp_xds_var_double(mat_final_filt,nplr,naz,nspec,iA(1),iA(2),iA(3),nbins,bc,imtype);
        end
        mat_recFXD=reshape(mat_recFXD,nX,nY,nZ,nB);         disp('Reshape output from mex-file');
    
    
    elseif bpCodeFlag==2  % MATLAB
        
        
        %%%%%%%%%%%%%% THIS IS WHAT WE OFTEN USE 
        mat_recFXD=bp4d_multistage(mat_final_filt,  lshift/orgnbins*nbins, pinterpflag, ifactor, imtype);
%         mat_recFXD=mat_recFXD/pi^3;  IT IS A BUG, ZHIWEI QIAO FOUND.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
    elseif bpCodeFlag==3  % C is just for single 
        bc=single(bc);imtype=single(imtype);
        nplr=single(nplr);naz=single(naz);nspec=single(nspec);
        iA=single(iA);nbins=single(nbins);
        
        mat_recFXD=bp_xds_single(mat_final_filt,nplr,naz,nspec,iA(1),iA(2),iA(3),nbins,bc,imtype);
%        mat_recFXD=reshape(mat_recFXD,nX,nY,nZ,nB);         disp('Reshape output from mex-file');
    else
        fprintf('Error: set bpCodeFlag to 1,2,3\n');
    end
end
t=cputime-t;
fprintf('\n cpu time for BackProj: %5.3f\n',t);