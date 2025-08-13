function [kernel] = ConvKernel( gridRes, alpha, W, kernelScale )
% parameters
%   gridRes - the size of the gridded data, NxN
%   [alpha=2] - the gridding factor (typically 2x)
%   [W=3] - convolution kernel size (e.g., 3)
%   [kernelScale=50] - kernel scale wrt gridRes (e.g., 50)
% output
%   kspace - the output kspace data

% check optional parameters
if ( nargin==1 )
    alpha = 2;
    W = 3;
    kernelScale=100;
elseif ( nargin==2)
    W = 3;
    kernelScale=100;
elseif (nargin==3)
    kernelScale=100;
end


% check inputs
if ( alpha < 1 )
    error('I will not knowingly undergrid');
end


% set the size of the grid for the kernel in grid-units
kernRes = W;%ceil(W)+2;
if ( mod(kernelScale*kernRes,2) == 0 )
    %error('kernel width should not be even');
    kernRes = kernRes+1;
end


% calculate the KB kernel
kx = linspace( -W/(2*gridRes), W/(2*gridRes), kernelScale*kernRes );
kc = ceil( (length(kx)+1)/2 );
beta = pi * sqrt( W^2/alpha^2 * (alpha - 0.5)^2 - 0.8 );
C_k = 1/W * besseli( 0, beta*sqrt(1-(2*gridRes*kx/W).^2) );
C_k = abs(C_k);

C_k2 = C_k' * C_k;
C_k2 = C_k2 / max(C_k2(:));


% get apod kernel
x = linspace( -0.5, 0.5, gridRes );
x = linspace( -(2*gridRes)/W, (2*gridRes)/W, gridRes );
x = linspace( -gridRes/2, gridRes/2, gridRes );
c_k = sin( sqrt( (pi*W*x/gridRes).^2 - beta^2 ) ) ./ sqrt( (pi*W*x/gridRes).^2 - beta^2 );
c_k2 = c_k' * c_k;


kernel.kernel = C_k2;
kernel.apod = c_k2;
kernel.W = W;
kernel.alpha = alpha;
kernel.gridRes = gridRes;
kernel.kernRes = kernRes;
kernel.kernelScale = size(C_k2,1);