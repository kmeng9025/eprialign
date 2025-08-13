function [dens1D dens] = ConvGridDensityHJ( gridX, gridY, gridRes, kernel )
% parameters
%   gridX - the kx coordinates for the data
%   gridY - the ky corrdinates for the data
%   gridRes - the size of the gridded data, NxN
%   kernel - kernel struct from ConvKernel
% output
%   kspace - the output kspace data

% check inputs
numPoints = length(gridX);
if ( length(gridY) ~= numPoints )
    error('invalid grid inputs, gridX and gridY must be the same size\n');
end

% mask the image to prevent noisy density functions
%[currConv] = ConvGrid( ones(numPoints,1), gridX, gridY, gridRes, ones(gridRes), kernel );
%mask = currConv > 1e-3;
% mask = currConv > 1e6;
%keyboard

% density compensation
lastSum = 0;
dens1D = ones(numPoints,1);
dens = ones(gridRes,gridRes);
% dens1D = zeros(numPoints,1);
lastdens1D = dens1D;

% currConv = real(LibConvGrid( dens+j, gridX, gridY,...
%     gridRes, ones(gridRes, gridRes), kernel.kernel, kernel.kernRes ));
% dens = ones(gridRes,gridRes) ./ currConv; 
% figure(11),
% imagesc( real(dens) ); axis square; drawnow;

for nits=1:100

    % Grid density function    
    currConv = real(LibConvGrid( dens1D+j, gridX(:), gridY(:),...
        gridRes, ones(gridRes, gridRes), kernel.kernel, kernel.kernRes ));

    % Extract density value from gridded density map
    % (Needed for non-cartesian sampling)    
    currDens1D = ones(numPoints,1);
    for i=1:numPoints
        x = gridX(i);
        y = gridY(i);
        
        % bi-linear interpolation 
        tmpQX = ceil(x)-floor(x);
        tmpQY = ceil(y)-floor(y);
        tmpQ  = tmpQX * tmpQY;
        tmpX1 = ceil(x)-x;
        tmpX2 = x-floor(x);
        tmpY1 = ceil(y)-y;
        tmpY2 = y-floor(y);
        if ( (tmpQX == tmpQY) && (tmpQ == 0 ) ) % no interpolation needed
            intDens = currConv( y, x );
        elseif ( tmpQX == 0 ) % interp in Y 
            intDens = tmpY1/tmpQY*currConv( floor(y), floor(x) ) + ...
                        tmpY2/tmpQY*currConv( ceil(y), floor(x) );
        elseif ( tmpQY == 0 ) % interp in X
            intDens = tmpX1/tmpQX*currConv( floor(y), floor(x) ) + ...
                        tmpX2/tmpQX*currConv( floor(y), ceil(x) );
        else % interp in X+Y
            intDens = currConv( floor(y), floor(x) ) / tmpQ * tmpX1 * tmpY1 + ...
                    currConv( floor(y), ceil(x)  ) / tmpQ * tmpX2 * tmpY1 + ...
                    currConv( ceil(y), floor(x)  ) / tmpQ * tmpX1 * tmpY2 + ...
                    currConv( ceil(y), ceil(x)   ) / tmpQ * tmpX2 * tmpY2;
        end   
        currDens1D(i) = intDens;
    end
    
    dens1D = dens1D ./ real(currDens1D);
    dind = isnan(dens1D); dens1D(dind) = 0;
    dind = isinf(dens1D); dens1D(dind) = 0;    
    
    % Stopping criterion    
    currDiff = abs(sum(dens1D(:) - lastdens1D(:)));
    if ( currDiff  < 1e-30 )
        break;
    else
        if nits > 1
            if currDiff > lastDiff 
                dens1D = lastdens1D;
                break;
            end  
        end
        lastdens1D = dens1D;
        lastDiff = currDiff;
        lastDens = dens;
    end
end
dens = real(dens);
dens1D = real(dens1D);
% fprintf('density compensation converged to %g after %g iterations\n',currDiff,nits);
