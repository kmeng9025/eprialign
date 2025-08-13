% FIT_LOOKUP1 - lookup dictionary fitting to a function with 1 parameters
% function [res, pars] = FIT_LOOKUP1(y, pars, what_to_do)
% y     - data array, M traces, N points each [double NxM]
% pars  - [structure] of parameters for dictionary generation   
%   [].func - Function of 3 arguments [f.e. @(x, A) exp(-x/A)]
%   [].par1 - Array of first argument values [double array]
%   [].x    - Array of function arguments [double array, Nx1]
%   [].ndict - Normalized dictionary array
%   [].normV - Norm of dictionary
% what_to_do  - Generate dictionary/fit data using dictionary/both [0 | 1 | 2]
% res   - Results [double array, Mx2]
% See also FIT_LOOKUP2, FIT_LOOKUP3
% 
% Example:
% y = rand(100, 16);
% pars.func = @(x, A) exp(-x/A);
% pars.par1 = linspace(0.3, 1.1, 200);
% pars.x  =  linspace(0, 10, 16)';
% [~, pars] = fit_lookup1(y, pars, 0);
% [res] = fit_lookup1(y, pars, 1);

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago,JULY 2013
% Contact: epri.uchicago.edu

function [fit, pars] = fit_lookup1(y, pars, what_to_do)
fit = [];

nx = length(pars.x);
x = pars.x(:);
ntr = size(y,2);
np1 = length(pars.par1);

% Generate dictionary
if what_to_do == 0 || what_to_do == 2
  dict = zeros(nx, np1);
  for ii=1:np1
      dict(:, ii) = pars.func(x, pars.par1(ii));
  end
  normV = sqrt(sum(dict.^2,1)); %Normalize final library vectors
  pars.ndict = dict ./ repmat(normV,nx,1);
  pars.normV = normV;
end

if what_to_do ~= 0
%   res = y'*pars.dict';
%   
%   [aaa, idx] = max(res);
%   
%   fit(2) = pars.par(idx);
%   fit(1) = mean(y ./ pars.func(pars.x, fit(2)));

  res = y'*pars.ndict;
  [~, idx] = max(res, [], 2);
  
  fit = zeros(ntr, 3);
  for jj=1:ntr
    fit(jj, 2) = pars.par1(idx(jj));
    amp = sum(y(:,jj) .* pars.ndict(:, idx(jj)));
    fit(jj, 1) = amp / pars.normV(idx(jj)); 
    fit(jj, 3) = norm(y(:,jj) - amp*pars.ndict(:, idx(jj)));
  end
end