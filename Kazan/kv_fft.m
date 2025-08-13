function [varargout] = kv_fft(ax, y, pars)
% FFT processing. Written for the FFTPLUGIN of Kazan Viewer
%     [rax, ry]= kv_fft(ax, y, pars)
%     [rax, ry, apwin]= kv_fft(ax, y, pars)
%     
%  Depending on the fields of 'pars' preprocess data for FFT
%  including data cutting, windowing and zerofilling and performs
%  1D FFT of 'y' first dimension. Some special tipes of FFT are supported.
%   'y' sould be column or 2D matrix.
%   'ax' has to be a structure with the field x (X axis).
%   'pars' has to be a structure. Possible fields:
%      pars.data  - _0, _0_, 0_ - the 0 time is at right/center/left
%      pars.opt   - real/imag/qseq/cta, type of FFT
%      pars.rshift, pars.lshift - cut data from the appropriate side
%      pars.awin   - apodization window, do not use +/-
%          'bla','bar','con', 'cos','ham','han','wel',
%          'exp','gau', 'kai',
%          'sq', 'none'
%      pars.awidth    - width of apo win in % to overall points
%      pars.ashift    - shift of apo win from the zero time pozition
%      pars.aalpha    - required by some apo windows
%      pars.awings    - off/on, cut data outside of apowindow
%      pars.zerofill  - zerofilling, 0 - add zeros to the closest number
%                       2^n, 1 - ... 2^(n+1) e.t.c.
%      pars.fft       - 0/1 FFT will not be performed/ will be performed
%      pars.phase0    - 0th order phase correction
%      pars.phase1    - 1st order phase correction (degree/MHz)
%      pars.xscale    - X axis scalling factor (1)
%
% See also FFT, FFTSHIFT, APOWIN, SAFEGET

% NOTE:
%     This function uses 'apowin' from EasySpin, and 'safeget' from
%     KazanViewer packages
% AlSi 26.09.2004
% boep 31.10.2006

x   = ax.x;
rax = ax;
ndim  = size(y,1);
ndim2 = size(y,2);

left_shift = 0;

fft_type = safeget(pars, 'opt', 'real');

switch fft_type
    case {'real', 'cta'}
        y = real(y);
end

% append data from the right
rshift = floor(safeget(pars, 'rshift', 0)+0.5);
if rshift > 0
    y1(rshift+1:ndim, :) = y(1:ndim-rshift, :);
    y1(1:rshift, :) = 0;
else
    %script{end+1} = sprintf('y1 = y;', rshift);
    y1 = y;
end

% cut data arrays from the left
lshift = floor(safeget(pars, 'lshift', 0)+.5);
if lshift > 0
    y1 = y1(lshift+1:end, :);
end
ndim1 = size(y1,1);

fft_data = safeget(pars, 'data', '0_');
switch fft_data
    % make array simmetric around the center
    case '_0_'
        center = safeget(pars, 'center', ndim/2);
        span = min(center - lshift, ndim - rshift - center);
        y1 = y1(center+[-span+1:span],:);
        left_shift = center - span;
        center = 0;
    case '_0'
        center = safeget(pars, 'center', ndim1/2);
    otherwise
        center = -ndim1/2;
        left_shift = lshift;
end

dx = diff(x);
dx = sum(dx)/length(dx);

ndim_new = size(y1,1);

% apodization window
awin = safeget(pars, 'awin', 'none');
if ~strcmp(awin, 'none')
    wings = safeget(pars, 'awings', 'on');
    width = abs(safeget(pars, 'awidth', .01));
    npoints = floor(ndim * width + 0.5);
    switch pars.awin
        case {'bla','bar','con', 'cos','ham','han','wel'}
            aw = apowin(pars.awin, npoints*2);
        case {'exp','gau', 'kai'}
            aw = apowin(pars.awin, npoints*2, safeget(pars, 'aalpha', 2));
        case {'sq'}
            aw = ones(npoints*2, 1);
            wings = 'off';
        otherwise
            aw = ones(npoints*2, 1);
    end
    if strcmp(wings, 'on')
      lastv = aw(end);
    else
      lastv = 0;
    end
    apo_shift = safeget(pars, 'ashift', 0);

    % make window twice larger
    awin = ones(ndim*4, 1)*lastv;
    awin(floor(ndim*2+center+apo_shift+(-npoints:npoints-1)+0.5),1) = aw;
    awin = awin(floor(ndim*2+(-ndim_new/2:ndim_new/2- 1)+0.5));
else
    awin = ones(ndim_new, 1);
end

if isfield(pars, 'awin')
    y1 = y1.*awin(:, ones(size(y1, 2), 1));
end

ww = zeros(ndim, 1);
ww(max(left_shift,0)+(1:ndim_new)) = awin;
varargout{3} = ww;
    
if safeget(pars,'fft',1) == 0
    rax.x = x;
    y = zeros(ndim, ndim2);
    y(max(left_shift,0)+(1:ndim_new),:) = y1;
    varargout{1} = rax;
    varargout{2} = y;
    return;
end

% clear y

%**************************************************************************
%**************************************************************************
%**************************************************************************

% zerofilling
ndim1 = ndim;
zfill = floor(safeget(pars, 'zerofill', 0)+0.5);
if zfill
    npow  = fix(log2(ndim)+.5); %AlSi (ceil -> fix)
    ndim1 = 2^(abs(zfill)+npow);
    yfill = zeros(ndim1, ndim2);
    switch fft_data
        case '0_'
            if ndim1 < ndim
                yfill = y1(1:ndim1, :);
            else
                yfill(1:ndim_new, :) = y1;
            end
        case '_0'
        case '_0_'
            yfill(round((ndim1-ndim_new)/2)+(1:ndim_new),:) = y1;
    end
    y1 = yfill;
    clear yfill
else
%     switch fft_data
%         case '0_'
%     end
end
ndim_new = size(y1,1);

switch fft_type
    case 'qseq'
        y1(2:2:end) = -y1(2:2:end);
        ytmp=zeros(1,2*ndim1);
        ytmp(1:2:2*ndim1)=real(y1);
        ytmp(2:2:2*ndim1)=imag(y1);
        ry = fft(real(ytmp));
        rx = [-ndim1:ndim1-1].'*1/dx/ndim1/2;
    case 'real'
        rx = [0:ndim1/2-1].'*1/dx/ndim1;
        ry = fftshift(fft(y1), 1);
        ry = ry(ndim1/2+1:end, :);
    case 'cta', % cross-term averaging
        avpar = safeget(pars, 'cta', ndim/4);
        rx = [0:ndim1/2-1].'*1/dx/ndim1;
        ry = 0;
        for p=1:avpar,yy=y1(p:end,:);
            yy(end+1:ndim1,:)=0;
            ry = ry + abs(fftshift(fft(yy), 1)).^2;
        end
        ry = sqrt(ry(ndim1/2+1:end, :));
    case '2ch'
        rx = (0:ndim1/2-1)'*1/dx/ndim1;
        ry = abs(fftshift(fft(real(y1)), 1)) + ...
          1i*abs(fftshift(fft(imag(y1)), 1));
        ry = ry(ndim1/2+1:end, :);
  otherwise
        rx = [-ndim_new/2:ndim_new/2-1].'*1/dx/ndim1;
        switch fft_data
            case '0_'
                ry = fftshift(ifft(y1), 1);
            case '_0'
                ry = fftshift(ifft(y1), 1);
            case '_0_'
                ry = fftshift(ifft(ifftshift(y1,1)), 1);
        end
end
xscale = safeget(pars, 'xscale',1);
xshift = safeget(pars, 'xshift',0);
rx = rx * xscale + xshift;

[rax.xlabel,rax.x] = findunit(safeget(ax, 'xlabel', ',s'), rx);

% if isfield(pars, 'CorrectBaseline')
  sh = fix(size(ry, 1) / 16);
  mn = mean(mean(real(ry([1:sh,end-sh:sh], :))));
  ry = ry - mn;
% end

if isfield(pars, 'FOV')
    FOV = pars.FOV;
    cut = find(abs(rax.x) < FOV/2);
    rax.x = rax.x(cut);
    ry  = ry(cut,:);
end


if isfield(pars, 'phase0')
    ry = ry.*exp(-1i*pars.phase0*pi/180);
end
if isfield(pars, 'phase1')
    phase = exp(-1i*pars.phase1*pi/180*rax.x);
    ry = ry.*phase(:, ones(size(ry,2),1));
end

varargout{1} = rax;
varargout{2} = ry;
    
%-----------------------------------------------------
function [newlabel,newaxx] = findunit(label, axx)
newlabel = label;
newaxx = axx;
p = findstr(label, ',');
if isempty(p), return; end
[un, unit, c1, c2] = kvgetvalue(['1',label(p+1:end)]);
[a,b,c] = kv_bestunit(max(abs(axx))/un, '');
newaxx = axx/(c*c2);

switch unit
   case 's',  unit = 'Hz'; name = 'Frequency,';
   case 'Hz',  unit = 's'; name = 'Time,';
   otherwise, unit = ''; name = ['1/',label(1:p)];
end
newlabel = [name, ' ', b, unit];

  
