function [ ds ,fn , h_center,pars_out,immask,threshold] = cw_get_img(mat_fn,data_fn,threshold)
% [ ds ,fn ,h_center,pars_out,immask ] = get_img(mat_fn,data_fn,pars_out)
% mat_fn is the mat file name as a string
% data_fn is the data array variable name as a string
%  data_fn variable is typically named: mat_recFXD and
%  contains typically 64 x 64 x 64 x 64        points
%                  spatial x      y      z      epr spectrum
% limit= .15 of maximum signal in image cm 00/12/07
% ds is the set of data sets as x,y column pairs [x1 y1 x2 y2 ...]
% fn is the list of dsnn files that are in ds
% h_center are the center field values
% pars_out contains the spectral data etc. data from the header file
% immask contains the non-zero elements of the image
% modification to immask = contains true intensity info not 1 or 0
%  get threshold value from cpv list to cut off low intensity spectra
%this version changed on February 217 2004 - sqrt(2) by CM  sqrt 2 multiplication to get
% field axis correct

% mat_recFXD was hard coded rather than image_data
% KA and CH replaced all hard coded mat_recFXD with image_data

% load the data mat file with all the  data arrays and variables
%keyboard
eval([ 'load ' mat_fn ])
if exist('pars_out', 'var') 	%  use input pars-out file  from saved data in mat-file    added by cm 120203
  disp('pars_out from saved mat file')
else                        % or read the parameter file  from raw data on eprima or bruker for the reconstruction program
  disp('pars_out does not exist in the mat file')
  if exist('name_com', 'var'), prefix=eval('name_com'), else
    prefix=input('enter directory and name of raw image data on acquisition computer ','s')
  end
  pars_out=read_ascii_header(prefix);  % read the parameter file  from raw data for the reconstruction program
  disp(strcat('pars_out from ' ,prefix))
end
if exist('q') && isfield(q,'com'), com=q.com;end
if exist('com'), isAbsorption = ~com(7); else isAbsorption = 1;end

disp('header read OK')

image_data= (eval(data_fn));    % read the image and make spectra
dataTypeStr=class(image_data);
%data_fn variable is typically named: mat_recFXD
%can contain  128 x 128 x 128 x 128 points
%            x     y    z    epr spectrum
% convert to one  file named ds which only
% has epr signal y-values as columns.
% get actual size of data set and make mask

jj=ndims(image_data) ;

%XB, YB or XB image 1 D spatial, 1D spectral
if jj == 2
  [a b ]=size(image_data) ;
  %immask=(zeros(a,1));
  % permute mat_fn to allow max to  work on correct dimensions
  %  threshold default was 1/2 of max
  %immask=max(permute(image_data,[3 1 2]))>max(max(max(image_data(:,:,:))))/2;
  %keyboard
  %immask=(max(image_data))>max(max(max(max(image_data))))*threshold;
  %immask=(max(image_data,[],2))>max(max(max(image_data,[],2)*threshold));
  immask=max(image_data,[],2);                            % find maxima of all spectra
  idx=find(immask<max(immask)*threshold);   % find indices of signals < threshold
  immask(idx)=0;                                                    % and set them to zero
  % immask now holds intensities all are > max*threshold
  %immask=squeeze(immask);
  %immask=fliplr(immask); % to get registration compatible with dection and area programs 3/7/01 cm
  % reorganise data set
  % make up ds_spec data set array before loop to speed up
  %recV=reshape(image_data,a,b);
  %recV=recV'; % spectra as columns
  %immaskV=reshape(immask,a,b);
  %ii=find(immaskV);
  %ds_spec=zeros(b,size(ii,1));
  %ds_spec=recV(:,ii); % spectra now in c x no of voxels formidx=find(immask);
  idx=find(immask);
  d1=size(idx,1);
  d2=size(image_data,2);
  ds_spec=zeros(d2,d1,dataTypeStr);
  set_gl('n_ds',d2);      %sets data set size
  m=1;
  for j=1:b ,j ;
    if immask(j)>0;
      temp=squeeze(image_data(j,:));
      %keyboard
      ds_spec(:,m)=temp';
      m=m+1;
    end
  end
end % end of jj = 2 loop


%XZB image 3D 2spatial, one spectral
if jj==3
  [a b c]=size(image_data) ;
  %immask=(zeros(a,b));
  % permute mat_fn to allow max to  work on correct dimensions
  %  threshold default was 1/2 of max
  %immask=max(permute(image_data,[3 1 2]))>max(max(max(image_data(:,:,:))))/2;
  %keyboard
  threshold=get_gl('threshold');
  if( isempty('threshold')),threshold= input('set threshold cutoff between 0 and 1, choice = ?');,end
  %immask=(max(image_data))>max(max(max(max(image_data))))*threshold;
  %immask=(max(image_data,[],3))>max(max(max(image_data,[],3)*threshold));
  immask=max(image_data,[],3);                            % find maxima of all spectra
  idx=find(immask<max(max(immask))*threshold);   % find indices of signals < threshold
  immask(idx)=0;                                                    % and set them to zero
  % immask now holds intensities all are > max*threshold

  %immask=squeeze(immask);
  %immask=fliplr(immask); % to get registration compatible with dection and area programs 3/7/01 cm
  % reorganise data set
  % make up ds_spec data set array before loop to speed up
  %recV=reshape(image_data,a*b,c);
  %recV=recV'; % spectra as columns
  %immaskV=reshape(immask,a*b,1);
  %ii=find(immaskV);
  %ds_spec=zeros(c,size(ii,1));
  %ds_spec=recV(:,ii); % spectra now in c x no of voxels form

  idx=find(immask);
  d1=size(idx,1);
  d2=size(image_data,3);
  ds_spec=zeros(d2,d1,dataTypeStr);
  m=1;
  for i=1:a , i
    for j=1:b
      if immask(i,j)>0;
        temp=squeeze(image_data(i,j,:));
        ds_spec(:,m)=temp;
        m=m+1;
      end
    end
  end  % end of jj=3 loop
end

if jj==4    % e.g. XYZB image 4D 3spatial, one spectral

  [a b c d]=size(image_data);
  %immask=(zeros(a,b,c));
  % permute mat_fn to allow max to  work on correct dimensions
  %  threshold get from data or else enter
  % check

  % continue
  %immask=(max(image_data,[],4))>max(max(max(max(image_data,[],4))))*threshold;
  % for integral of spectrum
  %immask=(sum(image_data,[],4))>max(max(max(sum(image_data,[],4))))*threshold;
  immask=max(image_data,[],4);                            % find maxima of all spectra
  idx=find(immask<max(max(max(immask)))*threshold);   % find indices of signals < threshold
  immask(idx)=0;                                                    % and set them to zero
  % immask now holds only intensities that all are > max*threshold

  %immask=squeeze(immask);
  % speed up code - size ds before loop
  %keyboard
  %recV=reshape(image_data,a*b*c,d);
  %recV=recV'; % spectra as columns
  %immaskV=reshape(immask,a*b*c,1);
  %ii=find(immaskV);
  %ds_spec=zeros(d,size(ii,1));
  %ds_spec=recV(:,ii); % spectra now in d x no of voxels form
  idx=find(immask);
  d1=size(idx,1);
  d2=size(image_data,4);
  ds_spec=zeros(d2,d1,dataTypeStr);
  % reorganise data set into spectra
  %keyboard
  m=1;
  for i=1:a
    for j=1:b
      for k=1:c
        if immask(i,j,k)>0;
          temp=squeeze(image_data(i,j,k,:)) ;
          ds_spec(:,m)=temp;
          m=m+1;
        end
      end
    end
  end
end  % end of jj==4 loop

disp('image data read OK')


[ N,p]=size(ds_spec);

if isAbsorption
  % take adjacent differences of ds_spec y values and build new dsdata
  % which is differential of the  absorption ds_spec to mimic CWEPR expt data
  dsdiff=diff(ds_spec); %   difference now only N-1 elements
  % add back one element at start equal to first difference%
  dsdiff=[dsdiff(1,:);dsdiff];
  dsdata=dsdiff; % now have differential form of the epr spectra
else
  dsdata=ds_spec;
end


VDN=pars_out(11);
if VDN == 18, calib = 15.788; % gauss/volt old magnet as of about 6\15\2002 onwards
elseif VDN == 9,  calib=1 ; % new intermediate magnet 4/8/2003;
  % now unit in gauss in Intermediate magnet!

elseif VDN == 8,  calib=15.64 ;  % old magnet old data

elseif VDN == 7,  calib=12.38; % old vdn's

elseif VDN == 6; calib=4.232; % old vdn's

elseif VDN == 0;  calib=10; % XBand
else disp('no valid voltage divider number found'),pause

end
% check on source of file - new or old - old has smaller pars_out than 23x1
[aa bb]=size(pars_out) ;
if (aa < 23)
  VDN == 8    % old software reports scan width in swu
  calib=15.64 ;  % old magnet old data ,
  scan_width=pars_out(19); % scan width in swu units
  scan_width=scan_width*calib; % scan width in gauss
else
  scan_width=pars_out(19); % scan width in gauss*sqrt(2) from Modula
end
%form x-axis
%keyboard
x  =  linspace(1,N ,N )';
dx =  scan_width/N;
x  =  dx *[0:(N -1)]';
cen=mean(x);
x=x-cen;
% fix for recon_gui scaling on imaging 1
% CH comment out divide by sqrt(2) (pars_out(19)=pars_out(19)/sqrt(2);)
% in read_ascii_header.m because
% recon_guit needs SW and length to be mult. by sqrt(2)
% get_img.m corrects for sqrt(2) for CW
% divide pars_out(19) above if necessary
% read_ascii_header_zoom.m divides both SW and length by sqrt(2)
% x=x*sqrt(2);
% end fix
[a b]=size(dsdata);
ds =zeros(a,2*b,dataTypeStr);
%add x-axis to ds
ds(:,1:2:(2*b))=x*ones(1,b,dataTypeStr);
%ds(a,1:2:(2*b))=repmat(x,b);   % faster I think
% keyboard
% add spectra to ds
ds(:,2:2:(2*b))=dsdata;
fn =[];
h_center =zeros(1,b,dataTypeStr) ;

disp([num2str(size(ds,2)/2) ' spectra formed'])
% don't go further don't use fn in image
% set up mask for running display of spectra chosen for fit
%xx=[1:1:b]';
%xx_b=num2str(xx);
%fn=['fn_'];
%fn_b=repmat(fn,b,1);
%fn=[fn_b xx_b];
%keyboard
%primitive=[1;zeros(99,1)];
%[xxm,index]=mask(xx,primitive); % pick every 100th element of ds
%clear xxm primitive xx
%for n=1:b
%    if  index(n)==1,
%        disp(['spectrum number ' num2str(n) ' of ' num2str(b)])
%    end
%      file=[ 'ds' num2str(n)];	% give x,y set a name dsn
%      fn = str2mat(fn,file);
%  end
%  	[ m,n]=size(fn);
%  fn = fn(2:m,:);				% contains list of data set filenames n
% e.g. ds1 ds2 ds3 ds4 ....
%keyboard