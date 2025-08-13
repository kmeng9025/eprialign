function [p] = recon_FilterProjections(p, Pars, Rec)
%
%---INPUT---
% p = matrix of projections in column vectors
% Pars = Image Parameter Structure
% Rec = Reconstruction Parameter Structure
%-----------
% 
%---OUTPUT--
% p = filtered projections
%-----------
%



% % Window Projections - not currently implemented
%   b = maxflat(96,'sym',0.85);
%   w = pi*linspace(-1,1,num.xi);
%   h = abs(freqz(b,1,w))';
%   h = h(:,ones(1,Pars.nProj));
%   p = p.*h;

  CutOff = Rec.CutOff * Rec.nBins / Pars.nBins;
% Get Filter
  filt = designFilter(Rec.Filter, Pars.nBins, CutOff, Pars.nDim);
  
% frequency domain filtering
  p = fft(p,length(filt));
  filt = filt(:,ones(1,Rec.nProj));
  p = p.*filt;                
  p = real(ifft(p)); 
  p(Pars.nBins+1:end,:) = [];
end

%======================================================================
function filt = designFilter(filter, len, d, n)
% Returns the Fourier Transform of the filter which will be used to filter the projections
%
% INPUT ARGS:   filter - either the string specifying the filter 
%               len    - the length of the projections
%               d      - the fraction of frequencies below the nyquist which we want to pass
%               n      - number of dimensions
%
% OUTPUT ARGS:  filt   - the filter to use on the projections


  order = max(64,2^nextpow2(2*len));
  morder = 32*order;
  filt = 2*( 0:(morder/2) )./morder;
  filt = filt.^(n-1);
  w = 2*pi*(0:size(filt,2)-1)/morder;   % frequency axis up to Nyquist 

  switch filter
    case 'ram-lak'                  % No noise filtering
      % Do nothing             
    case 'butterworth'                % Butterworth Filter
      pow = 2;
      filt(2:end) = filt(2:end) .* (1 ./ (1 + (w(2:end)./(pi*d)).^(2*pow)));
    case 'wiener'                     % Wiener Filter
      [ds] = GetSpectrum('OX063H',0.0107,0,deltaH,len,1);
      ds(morder,2) = 0;
      Y = fft(ds(:,2));
      Y = Y ./ max(Y);
      Wien = abs(Y) ./ (abs(Y).^2 + N);
      filt(2:end) = filt(2:end).*Wien(2:end);
    case 'shepp-logan'
      % be careful not to divide by 0:
      filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
    case 'cosine'
      filt(2:end) = filt(2:end) .* cos(w(2:end)/(2*d));
    case 'hamming'
      filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d));
    case 'hann'
      filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2;
  end
  
  
  filt(w>pi*d) = 0;                      % Crop the frequency response
  filt = [filt' ; filt(end-1:-1:2)'];    % Symmetry of the filter
  filt = ifft(filt);
  filt(order/2+1:end-order/2) = [];
  filt = real(fft(filt));
end