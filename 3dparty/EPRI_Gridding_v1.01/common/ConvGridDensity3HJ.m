function [dens1D dens] = ConvGridDensity3HJ( gridX, gridY, gridZ, gridRes, kernel, nPar )
% parameters
%   gridX - the kx coordinates for the data
%   gridY - the ky corrdinates for the data
%   gridZ - the kz corrdinates for the data
%   gridRes - the size of the gridded data, NxN
%   kernel - kernel struct from ConvKernel
% output
%   dens - the output dunsity map
%   dens1D - density corresponding to sample array

if ~exist('nPar')
    nPar = 1;
end

% check inputs
numPoints = length(gridX);
if ( length(gridY) ~= numPoints )
    error('invalid grid inputs, gridX and gridY must be the same size\n');
end


% currConvity compensation
dens1D = ones(numPoints,1);
lastdens1D = dens1D;

for nits=1:3

    % Grid density function
    if nPar > 1
        currConv = real(LibConvGrid3Par( dens1D(:)+j, gridX(:), gridY(:), gridZ(:),...
            gridRes, ones(gridRes, gridRes, gridRes), kernel.kernel, kernel.kernRes, nPar )); 
    else
        currConv = real(LibConvGrid3( dens1D(:)+j, gridX(:), gridY(:), gridZ(:),...
            gridRes, ones(gridRes, gridRes, gridRes), kernel.kernel, kernel.kernRes ));  
    end
    
    % Extract density value from gridded density map
    % (Needed for non-cartesian sampling)
    currDens1D = ones(numPoints,1);
    for i=1:numPoints
        x = gridX(i);
        y = gridY(i);
        z = gridZ(i);
        
        if ceil(y) > size(currConv,1) || ceil(x) > size(currConv,2) || ceil(z) > size(currConv,3) ...
           || floor(y) < 1 || floor(x) < 1 || floor(z) < 1
           continue;
        end
        
        % bi-linear interpolation         
        tmpQX = ceil(x)-floor(x);
        tmpQY = ceil(y)-floor(y);
        tmpQZ = ceil(z)-floor(z);        
        tmpQ  = tmpQX * tmpQY * tmpQZ;
        tmpX1 = ceil(x)-x;
        tmpX2 = x-floor(x);
        tmpY1 = ceil(y)-y;
        tmpY2 = y-floor(y);
        tmpZ1 = ceil(z)-z;
        tmpZ2 = z-floor(z);        
        
        if ( tmpQX == 0 && tmpQY == 0 && tmpQZ == 0 ) % no interpolation needed
            intcurrConv = currConv( y, x, z );
        elseif ( tmpQX == 0 && tmpQZ == 0 ) % interp in Y 
            intcurrConv = tmpY1/tmpQY*currConv( floor(y), floor(x), floor(z)) + ...
                        tmpY2/tmpQY*currConv( ceil(y), floor(x), floor(z) );
        elseif ( tmpQY == 0 && tmpQZ == 0) % interp in X
            intcurrConv = tmpX1/tmpQX*currConv( floor(y), floor(x), floor(z) ) + ...
                        tmpX2/tmpQX*currConv( floor(y), ceil(x), floor(z) );
        elseif ( tmpQX == 0 && tmpQY == 0) % interp in Z
            intcurrConv = tmpZ1/tmpQZ*currConv( floor(y), floor(x), floor(z) ) + ...
                        tmpZ2/tmpQZ*currConv( floor(y), floor(x), ceil(z) );
        elseif ( tmpQX ==0 ) % interp in Y+Z
            intcurrConv = currConv( floor(y), floor(x), floor(z) ) / (tmpQY * tmpQZ) * tmpY1 * tmpZ1 + ...
                        currConv( floor(y), floor(x), ceil(z) ) / (tmpQY * tmpQZ) * tmpY1 * tmpZ2 + ...
                        currConv( ceil(y), floor(x), floor(z) ) / (tmpQY * tmpQZ) * tmpY2 * tmpZ1 + ...
                        currConv( ceil(y), floor(x), ceil(z) ) / (tmpQY * tmpQZ) * tmpY2 * tmpZ2;
        elseif ( tmpQY ==0 ) % interp in X+Z
            intcurrConv = currConv( floor(y), floor(x), floor(z) ) / (tmpQX * tmpQZ) * tmpX1 * tmpZ1 + ...
                        currConv( floor(y), floor(x), ceil(z) ) / (tmpQX * tmpQZ) * tmpX1 * tmpZ2 + ...
                        currConv( floor(y), ceil(x), floor(z) ) / (tmpQX * tmpQZ) * tmpX2 * tmpZ1 + ...
                        currConv( floor(y), ceil(x), ceil(z) ) / (tmpQX * tmpQZ) * tmpX2 * tmpZ2;
        elseif ( tmpQZ ==0 ) % interp in X+Y
            intcurrConv = currConv( floor(y), floor(x), floor(z) ) / (tmpQY * tmpQX) * tmpY1 * tmpX1 + ...
                        currConv( floor(y), ceil(x), floor(z) ) / (tmpQY * tmpQX) * tmpY1 * tmpX2 + ...
                        currConv( ceil(y), floor(x), floor(z) ) / (tmpQY * tmpQX) * tmpY2 * tmpX1 + ...
                        currConv( ceil(y), ceil(x), floor(z) ) / (tmpQY * tmpQX) * tmpY2 * tmpX2;
        else % interp in X+Y+Z
            intcurrConv = currConv( floor(y), floor(x), floor(z) ) / tmpQ * tmpY1 * tmpX1 * tmpZ1 + ...
                        currConv( floor(y), floor(x), ceil(z) ) / tmpQ * tmpY1 * tmpX1 * tmpZ2 + ...
                        currConv( floor(y), ceil(x), floor(z) ) / tmpQ * tmpY1 * tmpX2 * tmpZ1 + ...
                        currConv( floor(y), ceil(x), ceil(z) ) / tmpQ * tmpY1 * tmpX2 * tmpZ2 + ...
                        currConv( ceil(y), floor(x), floor(z) ) / tmpQ * tmpY2 * tmpX1 * tmpZ1 + ...
                        currConv( ceil(y), floor(x), ceil(z) ) / tmpQ * tmpY2 * tmpX1 * tmpZ2 + ...
                        currConv( ceil(y), ceil(x), floor(z) ) / tmpQ * tmpY2 * tmpX2 * tmpZ1 + ...
                        currConv( ceil(y), ceil(x), ceil(z) ) / tmpQ * tmpY2 * tmpX2 * tmpZ2;
        end
        currDens1D(i) = intcurrConv;
    end
    
    dens1D = dens1D ./ real(currDens1D);
    dind = isnan(dens1D); dens1D(dind) = 0;
    dind = isinf(dens1D); dens1D(dind) = 0;     
    
    % Stopping criterion
    currDiff = sum(abs(dens1D(:) - lastdens1D(:)));
    if ( currDiff  < 1e-30 )
        break;
    else
        if nits > 1
            if currDiff > lastDiff 
                currConv = lastcurrConv;
                break;
            end  
        end
        lastdens1D = dens1D;
        lastDiff = currDiff;
        lastcurrConv = currConv;
    end
end
dens = real(currConv);
dens1D = real(dens1D);

% fprintf('Density compensation converged to %g after %g iterations\n',currDiff,nits);
