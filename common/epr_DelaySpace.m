% [taus,TRs] = epr_DelaySpace(630e-9,2.4e-6,5,'log',14e-6,6e-6)
% methods:
%  lin_sat - delays linearly spaced; TRs have ESE saturation correction
%  log_sat - delays log spaced; TRs have ESE saturation correction
%  lin_lin - delays linearly spaced; TRs linearly correced
%  log_lin - delays log spaced; TRs linearly correced

function [taus, TRs] = epr_DelaySpace(firstpoint,lastpoint,n,method,varargin)
if nargin > 4
    TR = varargin{1};
else
TR = 26e-6;
end
if nargin > 5
    T1 = varargin{2};
else
T1 = 6e-6;
end
TRs = [];
switch method
    case {'linear', 'lin_sat'}
        taus = linspace(firstpoint,lastpoint,n);
        TRs = TR + T1*log((1-2*exp(taus/T1))/(1-2*exp(taus(1)/T1)));
    case {'log','log_sat'} 
        taus = 10.^(linspace(log10(firstpoint),log10(lastpoint),n));
        TRs = TR + T1*log((1-2*exp(taus/T1))/(1-2*exp(taus(1)/T1)));
    case 'lin_lin'
        taus = linspace(firstpoint,lastpoint,n);
        TRs = TR + taus;
    case 'log_lin'
        taus = 10.^(linspace(log10(firstpoint),log10(lastpoint),n));
        TRs = TR + taus;
end