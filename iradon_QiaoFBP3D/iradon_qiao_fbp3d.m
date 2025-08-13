
%**********************************************************************************
% Function: reconstruction software, which can reconstruct a 3-D object from projections
% Formula: 3-D inverse radon transform
% Implementation method: single stage, which is the standard back-projection method.
%***********************************************************************************
%
%
%
%***********************************************************************************
% object: the 3-D reconstructed object, is a 3-D array. the number of every dimension should be the same. For example it's size is [128 128 128]
% p: a 2-D array, row number=points number of every projection; colum number= number of projections
% radon_pars.x: unit direction vector's x coordination,a 1-D array,whose length is the number of the projections
% radon_pars.y: unit direction vector's y coordination,a 1-D array,whose length is the number of the projections
% radon_pars.z: unit direction vector's z coordination,a 1-D array,whose length is the number of the projections
% radon_pars.w: the weighting factors of all the projections,a 1-D array, whos length is the number of the projections
% radon_pars.size: the length(cm) of the projection
% recon_pars.nBins: the length of the side of the cube. For example, the final image is 128*128*128, then nBins=128
% recon_pars.FilterCutOff: the bandwidth of the using low-pass filter. the range is [0 1], 0 means all the frequency components are blocked. 1 means all pass.
% recon_pars.size: the length(cm) of the side of the cube. For example, the final image is 5cm*5cm*5cm, then lengthBins=5
% recon_pars.display: 0 means to not displaying the intermediate result; 1 means to display the result
% recon_pars.processor: 1 is CPU; 2 is GPU
% recon_pars.Filter: 1 is 2rd derivative; 2 is using 3-points derivative method; 3 is using two S-L Ramp filter.
% recon_pars.interp_method: 0 is zero-rank interpolation method; 1 is linear interpolation method. 2 is spline interpolation method. Note that for GPU method, you can just select 0 or 1;
% recon_pars.tasksliced: (GPU ONLY )1 means that the back projction process is divided  into many small tasks; 0 means there just is one whole task.
%**********************************************************************************


%***********************************************************************************
% Author: ZHIWEI QIAO
% Center for EPR imaging in vivo physiology
% University of Chicago, JULY,2013
% Contact: zqiao1@uchicago.edu
%***********************************************************************************


function object=iradon_qiao_fbp3d(P, radon_pars, recon_pars)

tic

GX=radon_pars.G(:,1);
GY=radon_pars.G(:,2);
GZ=radon_pars.G(:,3);
wt_factor=radon_pars.w;
length_of_projection=radon_pars.size;    % the length of the projection
number_of_finalimage=recon_pars.nBins;   % the pixel number of the object
length_of_finalcube=recon_pars.size;     % the length of the object cube
Bandwidth_of_lowpassfilter=recon_pars.FilterCutOff;  %  the bandwidth of the lowpass filter.
display=recon_pars.display;              % Display or do not display the output figure

interp_method=recon_pars.Interpolation;    % select interpolation method.
tasksliced=recon_pars.tasksliced;          % select GPU TASK distributing method


%******************************************************************
%***********************  LET THE PROJECTION TO BE EVEN POINTS   **
%******************************************************************
[m,n]=size(P);
if mod(m,2)~=0
  my_projection2=P;
  P(2:m+1,:)=my_projection2;
end
[number_of_projection,number_of_angle]=size(P);
length_of_projection=length_of_projection+length_of_projection/m;
%******************************************************************
%**************     end end end end end end      ******************
%******************************************************************

tt=-(number_of_projection/2):number_of_projection/2-1;
tt=tt*(length_of_projection/number_of_projection);  %%%%%%%%%%%%%%%%%%%%%%%% THE spacial axis of the projection


%******************************************************************
%*****************DISPLAY THE INITIAL PROJECTION**************
if display > 0
  hdisp = figure;
  subplot(2,2,1); plot(tt,P);
  title('initial projections');
end
%**********************************************************************
%**********************************************************************



%******************************************************************
%***********************  LOW PASS FILTRATION   ********************
%******************************************************************
for ii=1:number_of_angle
  proj_data_fft_new(:,ii)=fftshift(fft(P(:,ii)));
end
lowpass_pointnum=round(number_of_projection/2*Bandwidth_of_lowpassfilter);
proj_data_fft_new(1:(number_of_projection/2+1-lowpass_pointnum-1),:)=0;%NOW THE BANDWITH GOTTEN BY MY EYE, LATER USING AUTOMATICAL METHOD
proj_data_fft_new((number_of_projection/2+1+lowpass_pointnum):number_of_projection,:)=0;
my_projection_after_lowpass=zeros(number_of_projection,number_of_angle);
for ii=1:number_of_angle
  my_projection_after_lowpass(:,ii)=real(ifft(fftshift(proj_data_fft_new(:,ii))));%note that we should use REAL, NOT abs.
  
end
%******************************************************************
%**************     end end end end end end      ******************
%******************************************************************


%******************************************************************
%*****************DISPLAY THE NOISE REDUCED PROJECTION**************
if display > 0
  figure(hdisp);
  subplot(2,2,2); plot(tt,my_projection_after_lowpass);
  title('projections after noise reduction');
end
%**********************************************************************
%**********************************************************************


%******************************************************************
%    FILTER THE PROJECTIONS BY USING 6 different methods************
%******************************************************************


switch(recon_pars.Filter)
  case 1 %%%%%%%%%%  the first method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    proj_data_filtered=diff(my_projection_after_lowpass,2,1);
    proj_data_filtered(number_of_projection-1:number_of_projection,:)=0;
    proj_data_filtered=proj_data_filtered/(length_of_projection/number_of_projection).^2;
    %%%%%%%%% end end end end   end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
  case 2
    
    %%%%%%%%%%%%%  the second method   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    proj_data_filtered1=zeros(number_of_projection,number_of_angle);
    delta=length_of_projection/number_of_projection;
    for n=1:number_of_angle
      for m=1:number_of_projection
        
        if m==1
          proj_data_filtered1(m,n)=(-3*my_projection_after_lowpass(m,n)+4*my_projection_after_lowpass(m+1,n)-my_projection_after_lowpass(m+2,n))/2/delta;
        elseif m==number_of_projection
          proj_data_filtered1(m,n)=(my_projection_after_lowpass(m-2,n)-4*my_projection_after_lowpass(m-1,n)+3*my_projection_after_lowpass(m,n))/2/delta;
        else
          proj_data_filtered1(m,n)=(my_projection_after_lowpass(m+1,n)-my_projection_after_lowpass(m-1,n))/2/delta;
          
        end
      end
    end
    
    
    proj_data_filtered=zeros(number_of_projection,number_of_angle);
    
    
    for n=1:number_of_angle
      for m=1:number_of_projection
        
        if m==1
          proj_data_filtered(m,n)=(-3*proj_data_filtered1(m,n)+4*proj_data_filtered1(m+1,n)-proj_data_filtered1(m+2,n))/2/delta;
        elseif m==number_of_projection
          proj_data_filtered(m,n)=(proj_data_filtered1(m-2,n)-4*proj_data_filtered1(m-1,n)+3*proj_data_filtered1(m,n))/2/delta;
        else
          proj_data_filtered(m,n)=(proj_data_filtered1(m+1,n)-proj_data_filtered1(m-1,n))/2/delta;
          
        end
      end
    end
    
    %%%%%%%%%%%%%% END of the second method %%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  case 3
    %%%%%%%%%%%%%  the third method   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    proj_data_filtered1=zeros(number_of_projection,number_of_angle);
    delta=length_of_projection/number_of_projection;
    
    
    for n=1:number_of_angle   %%%% the first 5-points filtration
      for m=1:number_of_projection
        if m>=3&&m<=number_of_projection-2
          proj_data_filtered1(m,n)=(-my_projection_after_lowpass(m+2,n)+8*my_projection_after_lowpass(m+1,n)-8*my_projection_after_lowpass(m-1,n)+my_projection_after_lowpass(m-2,n))/12/delta;
        end
        
      end
    end
    
    
    proj_data_filtered=zeros(number_of_projection,number_of_angle);
    
    
    for n=1:number_of_angle   %%%% the second 5-points filtration
      for m=1:number_of_projection
        if m>=3&&m<=number_of_projection-2
          proj_data_filtered(m,n)=(-proj_data_filtered1(m+2,n)+8*proj_data_filtered1(m+1,n)-8*proj_data_filtered1(m-1,n)+proj_data_filtered1(m-2,n))/12/delta;
        end
        
      end
    end
    %%%%%%%%%%%%%%%  END of the 3th method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  case 4 %%%%%%%%%%%%%%%%%% THE 4th METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%  RL filter
    d=length_of_projection/number_of_projection;
    proj_data_filtered=zeros(number_of_projection,number_of_angle);
    mm=number_of_angle;
    nn=number_of_projection;
    
    
    
    
    for n=-(nn-1):(nn-1); % calculate the Unit Impulse Response of QIAO's filter
      if n==0
        h_qiao(n+nn)=1/12/d/d/d;
      elseif mod(n,2)==0
        h_qiao(n+nn)=1/2/pi/pi/n/n/d/d/d;
      else
        h_qiao(n+nn)=-1/2/pi/pi/n/n/d/d/d;
      end
    end
    
    for m=1:mm     %  the convolution process using QIAO's filter
      for n=1:nn
        
        for k=1:nn
          proj_data_filtered(n,m)=my_projection_after_lowpass(k,m)*h_qiao(nn+n-k)+proj_data_filtered(n,m);
        end
      end
      
    end
    
    proj_data_filtered=proj_data_filtered*d;%adjust the factor
    
    proj_data_filtered=-proj_data_filtered*4*pi*pi;%%%%%%%%%  ADJUST THE VALUE %%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%  END of the 4th  method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  case 5
    
    %%%%%%%%%%%%%%%%%% THE 5th METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%  SL filter %%%%%
    d=length_of_projection/number_of_projection;
    proj_data_filtered=zeros(number_of_projection,number_of_angle);
    mm=number_of_angle;
    nn=number_of_projection;
    
    
    
    
    n=-(nn-1):(nn-1); % calculate the Unit Impulse Response of QIAO's filter
    
    hsl_qiao=(1/(pi^3*d^3))*(1./(2*n+1).^2+1./(2*n-1).^2).*sin(pi*0.5*(2*n+1));
    
    for m=1:mm     %  the convolution process using QIAO's filter
      for n=1:nn
        
        for k=1:nn
          proj_data_filtered(n,m)=my_projection_after_lowpass(k,m)*hsl_qiao(nn+n-k)+proj_data_filtered(n,m);
        end
      end
      
    end
    
    proj_data_filtered=proj_data_filtered*d;%adjust the factor
    
    proj_data_filtered=-proj_data_filtered*4*pi*pi;%%%%%%%%%  ADJUST THE VALUE %%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%  END of the 5th  method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
  otherwise
    
    %%%%%%%%%%%%%%%%%% THE 6th METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%  hamming method  %%%%%%%%%%%%%%
    d=length_of_projection/number_of_projection;
    proj_data_filtered=zeros(number_of_projection,number_of_angle);
    mm=number_of_angle;
    nn=number_of_projection;
    
    
    
    
    for n=-(nn-1):(nn-1); % calculate the Unit Impulse Response of QIAO's hamming window filter
      if n==0
        h_qiao1(n+nn)=1/12/d/d/d;
      elseif mod(n,2)==0
        h_qiao1(n+nn)=1/2/pi/pi/n/n/d/d/d;
      else
        h_qiao1(n+nn)=-1/2/pi/pi/n/n/d/d/d;% h_qiao1 is the 0.54 item
      end
    end
    
    n=-(nn-1):(nn-1); % calculate the 0.46 item
    
    h_qiao2=cos((n+1)*pi)./(4*pi^2.*(n+1).^2*d^3)+cos((n-1)*pi)./(4*pi^2.*(n-1).^2*d^3);
    
    h_qiao2(nn+1)=1/24/d^3+1/16/d^3/pi^2;
    h_qiao2(nn-1)=1/24/d^3+1/16/d^3/pi^2;
    
    
    h_hamming=0.54*h_qiao1+0.46*h_qiao2;
    
    
    for m=1:mm     %  the convolution process using QIAO's filter
      for n=1:nn
        
        for k=1:nn
          proj_data_filtered(n,m)=my_projection_after_lowpass(k,m)*h_hamming(nn+n-k)+proj_data_filtered(n,m);
        end
      end
      
    end
    
    proj_data_filtered=proj_data_filtered*d;%adjust the factor
    
    proj_data_filtered=-proj_data_filtered*4*pi*pi;%%%%%%%%%  ADJUST THE VALUE %%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%  END of the 6th  method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%**********************************************************************
%**********************************************************************



%******************************************************************
%*****************DISPLAY THE FILTERED PROJECTIONS*****************
if display > 0
  figure(hdisp);
  subplot(2,2,3); plot(tt,proj_data_filtered);
  title('the Filtered Projection')
end
%**********************************************************************
%**********************************************************************


%**********************************************************************
%---------------DO WEIGHTING FOR THE FILTERED PROJECTION-------------
for ii=1:number_of_angle
  proj_data_filtered_wt(:,ii)=proj_data_filtered(:,ii)*wt_factor(ii);
end

%**********************************************************************
%**********************************************************************


%******************************************************************
%*****************DISPLAY THE WEIGHTED PROJECTIONS*****************

if display > 0
  figure(hdisp);
  subplot(2,2,4); plot(tt,proj_data_filtered_wt);
  title('the Filtered Projection after weighting')
end

%**********************************************************************
%**********************************************************************



%**********************************************************************
%--------------------BEGIN THE BACKPROJECTION-------------------------
%**********************************************************************

switch(safeget(recon_pars, 'CodeFlag', 'GPU'))
  
  case 'CPU'  %%%%%%%%%%%   CPU METHOD           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    disp('Single stage FBP 3D reconstruction using CPU.');
    object=zeros(number_of_finalimage,number_of_finalimage,number_of_finalimage);
    t=object;
    
    if interp_method==0;
      interp_string='*nearest';
    elseif interp_method==1;
      interp_string='*linear';
    else interp_string='*spline';
    end
    %   figure;
    %   tic;
    for ii=1:number_of_angle
      for x=1:number_of_finalimage
        for y=1:number_of_finalimage
          for z=1:number_of_finalimage
            xx=(x-number_of_finalimage/2)*length_of_finalcube/number_of_finalimage;
            yy=(y-number_of_finalimage/2)*length_of_finalcube/number_of_finalimage;
            zz=(z-number_of_finalimage/2)*length_of_finalcube/number_of_finalimage;
            
            t(x,y,z)=xx*GX(ii)+yy*GY(ii)+zz*GZ(ii);
          end
        end
      end
      
      object=object+interp1(tt,proj_data_filtered_wt(:,ii),t,interp_string,0);%key   key  key
      
      %     imagesc(squeeze(object(:,:,number_of_finalimage/2)));
      %     title(strcat('z slice NO: ',num2str(number_of_finalimage/2),': ',num2str(ii)));
      %     drawnow;
    end
    
    object=-1/4/pi/pi*object;
    
    %%%%%%%%%%%%%%%%%%%%%%%%   END OF CPU METHOD           %%%%%%%%%%%%%%%%%%%%
    
  case 'GPU'  %%%%%%%%%%%%%%%%%%%%%%%    BEGIN OF GPU METHOD  %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    disp('Single stage FBP 3D reconstruction using GPU.');
    object=zeros(number_of_finalimage,number_of_finalimage,number_of_finalimage);
    if tasksliced==1
      tic;
      k=parallel.gpu.CUDAKernel('EPRI_Kernel_slice.ptx', 'EPRI_Kernel_slice.cu');
      k.ThreadBlockSize=number_of_finalimage;
      k.GridSize=number_of_finalimage;
      for ii=1:number_of_finalimage
        object_ii=squeeze(object(:,:,ii));
        [object_GPU]=feval(k,object_ii,proj_data_filtered_wt,GX,GY,GZ,length_of_finalcube,number_of_finalimage,length_of_projection,number_of_angle,number_of_projection,ii,interp_method);
        
        object(:,:,ii)=gather(object_GPU);
        %            fprintf(strcat(num2str(ii),' the backprojection process of ZHIWEI GPU method need %f second\n'),toc);
      end
    else
      tic;
      k=parallel.gpu.CUDAKernel('EPRI_Kernel.ptx', 'EPRI_Kernel.cu');
      k.ThreadBlockSize=number_of_finalimage;
      k.GridSize=[number_of_finalimage number_of_finalimage];
      object_GPU=feval(k,object,proj_data_filtered_wt,GX,GY,GZ,length_of_finalcube,number_of_finalimage,length_of_projection,number_of_angle,number_of_projection,interp_method);
      object=gather(object_GPU);
    end
    
    
    %     fprintf('the backprojection process of ZHIWEI GPU method need %f second\n',toc);
    object=-1/4/pi/pi*object;
    %%%%%%%%%%%%%%%%%%%%%      END OF GPU METHOD  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
end
%**********************************************************************
%--------------------END  THE BACKPROJECTION--------------------------
%**********************************************************************

toc












