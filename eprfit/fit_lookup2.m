% FIT_LOOKUP2 - lookup dictionary fitting to a function with 2 parameters
% function [res, pars] = FIT_LOOKUP2(y, pars, what_to_do)
% y     - data array, M traces, N points each [double NxM]
% pars  - [structure] of parameters for dictionary generation   
%   [].func - Function of 3 arguments [f.e. @(x, A, B) (1 - A*exp(-x/B))]
%   [].par1 - Array of first argument values [double array]
%   [].par2 - Array of second argument values [double array]
%   [].x    - Array of function arguments [double array, Nx1]
%   [].ndict - Normalized dictionary array
%   [].normV - Norm of dictionary
% what_to_do  - Generate dictionary/fit data using dictionary/both [0 | 1 | 2]
% res   - Results [double array, Mx4]
%   1 - amplitude; 2 - parameter 1; 3 - parameter 2; 4 - fit error
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

function [fit, pars] = fit_lookup2(y, pars, what_to_do)
fit = [];

nx = length(pars.x);
x = pars.x(:);
ntr = size(y,2);
np1 = length(pars.par1);
np2 = length(pars.par2);

% Generate dictionary
if what_to_do == 0 || what_to_do == 2
  dict = zeros(nx, np1*np2);
  for ii=1:np1
    for jj=1:np2
      dict(:, sub2ind([np1, np2],ii,jj)) = pars.func(x, pars.par1(ii), pars.par2(jj));
    end
  end
  normV = sqrt(sum(dict.^2,1)); %Normalize final library vectors
  pars.ndict = dict ./ repmat(normV,nx,1);
  pars.normV = normV;
end

% Fit data
if what_to_do ~= 0
  
  % find size of the problem
  szy = size(y);
  dict_size  = numel(pars.ndict);
  max_traces = fix(18E9/dict_size/szy(1)); % tested on 8 GB RAM
  nsteps = ceil(szy(2)/max_traces);
  
  % break the problem into smaller pieces
  res = -1E24*ones(szy(2), nsteps);
  residx = zeros(szy(2), nsteps);
  for ii=1:nsteps
    nidx = (ii-1)*max_traces+1 : min(ii*max_traces, szy(2));
    [res(nidx, ii), residx(nidx, ii)] = max(y(:,nidx)'*pars.ndict, [], 2);
  end
  
  [~, idx] = max(res, [], 2);
%   [aaa, idx] = max(y'*pars.ndict, [], 2); % complete problem at once
 
  fit = zeros(ntr, 4);
  for jj=1:ntr
%     fidx = idx(jj);
    fidx = residx(jj,idx(jj));
    [i1,i2] = ind2sub([np1, np2],fidx);
    fit(jj, 3) = pars.par2(i2);
    fit(jj, 2) = pars.par1(i1);
    amp = sum(y(:,jj) .* pars.ndict(:, fidx));
    fit(jj, 1) = amp / pars.normV(fidx); 
    fit(jj, 4) = norm(y(:,jj) - amp*pars.ndict(:, fidx));
  end
end