% FIT_LOOKUP3 - lookup dictionary fitting to a function with 3 parameters
% function [res, pars] = FIT_LOOKUP3(y, pars, what_to_do)
% y     - data array, M traces, N points each [double NxM]
% pars  - [structure] of parameters for dictionary generation   
%   [].func - Function of 3 arguments [f.e. @(x, A, B, C) (1 - A*exp(-x/B))]
%   [].par1 - Array of first argument values [double array]
%   [].par2 - Array of second argument values [double array]
%   [].par3 - Array of second argument values [double array]
%   [].x    - Array of function arguments [double array, Nx1]
%   [].ndict - Normalized dictionary array
%   [].normV - Norm of dictionary
% what_to_do  - Generate dictionary/fit data using dictionary/both [0 | 1 | 2]
% res   - Results [double array, Mx3]
% See also FIT_LOOKUP1, FIT_LOOKUP3
% 
% Example:
% y = rand(100, 16);
% pars.func = @(x, A, B) (1 - A*exp(-x/B));
% pars.par1 = linspace(0.3, 1.1, 200);
% pars.par2 = linspace(1, 5, 200);
% pars.x  =  linspace(0, 10, 16)';
% [~, pars] = fit_lookup2(y, pars, 0);
% [res] = fit_lookup2(y, pars, 1);

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago,JULY 2013
% Contact: epri.uchicago.edu

function [fit, pars] = fit_lookup3(y, pars, what_to_do)
fit = [];

nx = length(pars.x);
x = pars.x(:);
ntr = size(y,2);
np1 = length(pars.par1);
np2 = length(pars.par2);
np3 = length(pars.par3);

% Generate dictionary
if what_to_do == 0 || what_to_do == 2
  pars.dict = zeros(nx, np1*np2*np3);
  for ii=1:np1
    for jj=1:np2
      for kk=1:np3
        dict(:, sub2ind([np1, np2, np3],ii,jj,kk)) = pars.func(x, pars.par1(ii), pars.par2(jj), pars.par3(kk));
      end
    end
  end
  normV = sqrt(sum(dict.^2,1)); %Normalize final library vectors
  pars.ndict = dict ./ repmat(normV,nx,1);
  pars.normV = normV;
end

% Fit data
if what_to_do ~= 0
  %   res = y'*pars.dict';
  %
  %   [aaa, idx] = max(res);
  %
  %   fit(2) = pars.par(idx);
  %   fit(1) = mean(y ./ pars.func(pars.x, fit(2)));
  
  fit = zeros(ntr, 3);
  our_par1 = pars.par1;
  our_par2 = pars.par2;
  our_par3 = pars.par3;
  
  % protection against memory overuse
  if numel(pars.ndict)*numel(y) > 1e9
    for jj=1:ntr
      res = y(:,jj)'*pars.ndict;     
      [aaa, idx] = max(res, [], 2);
      
      [i1,i2,i3] = ind2sub([np1, np2, np3],idx);
      fit(jj, 4) = our_par3(i3);
      fit(jj, 3) = our_par2(i2);
      fit(jj, 2) = our_par1(i1);
      amp = sum(y(:,jj) .* pars.ndict(:, idx));
      fit(jj, 1) = amp ;
      
%       figure(100); clf;
%       plot(x, y(:,jj), 'o'); hold on; 
%       plot(x , fit(jj, 1)*pars.ndict(:, idx));
%       text(0.1, 0.9, sprintf('%i XOVER=%4.2f LW=%4.2f PH=%4.2f', jj, fit(jj, 2), fit(jj, 3), fit(jj, 4)), 'units', 'normalized')
%       pause(0.5)
    end
  else
    res = y'*pars.ndict;
    
    [aaa, idx] = max(res, [], 2);
    
    for jj=1:ntr
      [i1,i2,i3] = ind2sub([np1, np2, np3],idx(jj));
      fit(jj, 4) = our_par3(i3);
      fit(jj, 3) = our_par2(i2);
      fit(jj, 2) = our_par1(i1);
      ppidx = abs(pars.dict(:, idx(jj))) > 0.5;
      fit(jj, 1) = mean(y(ppidx,jj) ./ pars.dict(ppidx, idx(jj)));
    end
  end
end