function  [pattern, resolution] = cw_shf(Spin_system, varargin)
% Spin_system [m x 4]  where m - number of nuclei and
%                  4 - [hyperfine,abundance,spin,equivalent nucs]
% varargin{1}          suggested resolution
% pattern              output pattern
% resolution           resolution

m = size(Spin_system, 1);
resolution = min(Spin_system(:,1))/2; 	% The smallest hfs /2 splitting resolution
%rest = min(s(:,1))/2; 	% original The smallest splitting resolution
if(nargin == 2)
   if(varargin{1} < resolution ), 
     resolution = resolution/round(resolution/varargin{1}); 
   end 
end

pattern = 1;
for ii=1:m
   pattern = cw_shf_add(pattern,resolution,Spin_system(ii,1),...
     Spin_system(ii,2), Spin_system(ii,3), Spin_system(ii,4));
end

pattern=pattern';

function  patn = cw_shf_add(pat,res,split,pct,spin,nequiv)
% new_pattern = shf_1(pattern,resolution,[ splitting,percentage,spin,numb_of_equivalent_spins])
% will update the pattern of shf or super hyperfine splitting
% The splitting (ie. abar) is the distance between successive lines 
% in the same units as the resolution: the distance between points.
% Spin is  particle's spin (e.g. 1/2 for H)
% and the percentage of the systems with that spin.
% and the number of equivalent spins.
% [ split pct spin nequiv]
% to insure complete accuracy of splitting pattern
% resolution <= split/2 is a real good idea
% the resolution must be <= the splitting
%if spin is 1/2 integer and either nequiv is odd or % < 100
% you will loose resolution if resolution = split.

% direct copy from shf_1, boep

fctn = pct/100;
ns = 2*spin;

% construct the pattern based on its own splitting
% assume the distance between points is splitting/2
% this guarantees that p1 is odd  being (2*ns+1) in length

p1 = [];
for k=1:ns
  p1 = [ p1 1 0 ];
end
p1 = [ p1 1 ] / (ns+1);

pad = zeros(size(p1));
pad(ns+1) = 1;

% pad is the same as p1 but is just a 1 in the center

% convolve it with itself nequiv times

patn = [ 1 ];

for k =1 : nequiv
  pat1 = conv(patn,p1);
  pad1 = conv(patn,pad);
  patn = fctn*pat1 + (1-fctn)*pad1;
end

% matlab code to linearly interpolate a stick spectrum
% Patn = = the pattern that is a stick pattern,
% on the x axis; which is at even internals given by  (split/2)
% unlike normal interpolation this assumes Patn=0 at all other places on x
LenP = length(patn);    % the length of Patn is an odd number
m = round( ( LenP-1)/2);
x = (-m:m)*(split/2);
% the new x axis xnew is also evenly space and symmetric and surrounds the
% old x axis.
m = ceil( x(LenP)/res);
xnew = (-m:m)*res;
%keyboard

% now find the bin closest to and below each of the stick places.
Y = zeros(size(xnew));
LenY = length(Y);
%disp( [ res split/2 ])
%keyboard
for k = 1:LenP
  bin = max( find(  xnew <= x(k) ) );
  f = (x(k) - xnew(bin) ) / res;
  % the fractional distance from the lower bin position
  %f=0
  Y(bin) = Y(bin) + (1-f)*patn(k);
  if(bin < LenY), Y(bin+1) = Y(bin+1) + f*patn(k);, end
end


patn = conv(Y,pat);

%keyboard


