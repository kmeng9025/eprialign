function mat_int_filt = FilterProjections(mat_int,imtype,filtflag,cutoff)
%function mat_int_filt = FilterProjections(mat_int,imtype,filtflag,cutoff);
%
% Last modified by KA dd061026
% recon_gui version 0.1

if filtflag==-1, mat_int_filt=mat_int;return;end

nplr=size(mat_int,4);
naz=size(mat_int,3);
nspec=size(mat_int,2);
nbins=size(mat_int,1);
%  FILTER THE PROJECTIONS
%  USING IMTYPE TO DETERMINE NUMBER OF FILTERING STEPS
mat_int_filt=mat_int;
[fbpfilt]=get_fbpfile(filtflag);
if imtype==1
    for n=1:2
        for i=1:nplr
            for m=1:naz
                mat_int_filt(:,:,m,i)=sino_filter(mat_int_filt(:,:,m,i),'ram-lak',1);
            end
        end
    end
end
if imtype==2|imtype==3|imtype==4|imtype==14
    for i=1:nplr
      for m=1:naz
        mat_int_filt(:,:,m,i)=sino_filter(mat_int_filt(:,:,m,i),'ram-lak',1);
      end
   end
end
for i=1:nplr
   for m=1:naz
      mat_int_filt(:,:,m,i)=sino_filter(mat_int_filt(:,:,m,i),fbpfilt,cutoff);
   end
end
% END OF function [mat_int_filt]=filter_xdsbp(mat_int,imtype,filtflag,cutoff);


%--------------------------------------------------------------------------
function [fbpfilt]=get_fbpfile(filtflag);

if filtflag==0
	fbpfilt='ram-lak';
elseif filtflag==1
	fbpfilt='shepp-logan';
elseif filtflag==2
	fbpfilt='cosine';
elseif filtflag==3
	fbpfilt='hamming';
elseif filtflag==4
	fbpfilt='hann';
else
	disp('INVALID FILTER');
	return;
end
% END OF function [fbpfilt]=get_fbpfile(filtflag);

%--------------------------------------------------------------------------
function output=sino_filter(p,filter,cutoff)
%'function sino_filter(spectra(nbins,nspectra),filter,cutoff)'

%Design the filter
len=size(p,1);
H = designFilter(filter, len, cutoff);
p(length(H),1)=0;  % Zero pad projections

% In the code below, I continuously reuse the array p so as to
% save memory.  This makes it harder to read, but the comments
% explain what is going on.

p = fft(p);    % p holds fft of projections

for i = 1:size(p,2)
   p(:,i) = p(:,i).*H; % frequency domain filtering
end

p = real(ifft(p));     % p is the filtered projections
p(len+1:end,:) = [];

output=p;
% END OF function output=sino_filter(p,filter,cutoff)

%--------------------------------------------------------------------------
function filt = designFilter(filter, len, d)
% Returns the Fourier Transform of the filter which will be
% used to filter the projections
%
% INPUT ARGS:   filter - either the string specifying the filter
%               len    - the length of the projections
%               d      - the fraction of frequencies below the nyquist
%                        which we want to pass
%
% OUTPUT ARGS:  filt   - the filter to use on the projections

order = max(64,2^nextpow2(2*len));

% First create a ramp filter - go up to the next highest
% power of 2.
%filt = 2*( 0:(order/2) )./order;

filt=zeros(order,1);
for i=2:2:order
   filt(i)=-1/(i-1-order/2)^2/pi^2;
end

filt(order/2+1)=1/4;
filt=real(fft(fftshift(filt)));
filt=filt(1:order/2+1)';
w = 2*pi*(0:size(filt,2)-1)/order;   % frequency axis up to Nyquist
%w = (2*pi*(0:size(filt,1)-1)/order)'; %frequency axis

switch filter
case 'ram-lak'
   % Do nothing
case 'shepp-logan'
   % be careful not to divide by 0:
   filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
case 'cosine'
   filt(2:end) = filt(2:end) .* cos(w(2:end)/(2*d));
case 'hamming'
   filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d));
case 'hann'
   filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2;   
otherwise
   error('Invalid filter selected.');
end

filt(w>pi*d) = 0;                      % Crop the frequency response
filt = [filt' ; filt(end-1:-1:2)'];    % Symmetry of the filter

% END OF function filt = designFilter(filter, len, d)