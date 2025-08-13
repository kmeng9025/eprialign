function [ gridImg, gridKSpace, gridDeapod ] = DoGriddingNonCarte( kSpace, gridParam, FOVscale )
% DoGridding with non-cartesian k-space samples

[gridKSpace, gridKDeapod] = GridKSpace(kSpace, gridParam, FOVscale);

if gridParam.dim==2
    gridDeapod = ifft2c( gridKDeapod);
    gridImg = ifft2c( gridKSpace );
    gridImg = gridImg ./ gridDeapod;
    gridImg = abm_center_crop_image( gridImg, [gridParam.N*gridParam.zp, gridParam.N*gridParam.zp] );
end

if gridParam.dim==3
    gridDeapod = ifftc(gridKDeapod);
    gridImg = ifftc( gridKSpace );
    gridImg = gridImg ./ gridDeapod;
    gridImg = abm_center_crop_image3( gridImg, [gridParam.N*gridParam.zp, gridParam.N*gridParam.zp, gridParam.N*gridParam.zp] );
end  
end


function [ kspaceOut, gridKDeapod ] = GridKSpace( kspaceIn, gridParam, FOVscale )
% GridKSpace regrid 2D or 3D k-data

% Parameter setting for gridding
dim = gridParam.dim;
N = gridParam.N;
zp = gridParam.zp;
alpha = gridParam.alpha;
gridW = gridParam.gridW;
gridScale = gridParam.gridScale;
traj = gridParam.traj;
nPar = gridParam.nPar;

if length(traj(:)) == 0 || length(kspaceIn(:)) == 0
    error('Error: Empty input data.');
end

% set grid size
gridAlpha = alpha;
gridSize = ceil( N*zp*gridAlpha);

% correct center-drifting
if mod(gridSize,2) == 0
    gridSize = gridSize+1;
end
gridCen = floor((gridSize+1)/2);

% 2D gridding
if dim==2
    gridX = traj(:,1);
    gridY = traj(:,2);

    gridKern = ConvKernel( gridSize, gridAlpha, gridW, gridScale );
    gridKDeapod = real (LibConvGrid( 1+j, gridCen, gridCen, gridSize, ones(gridSize), gridKern.kernel, gridKern.kernRes ) ); 

    currGridX = gridAlpha*gridX*FOVscale + gridCen;
    currGridY = gridAlpha*gridY*FOVscale + gridCen;
  
    dens1D = ConvGridDensityHJ( currGridX(:), currGridY(:), gridSize, gridKern );
    
    if exist('gridParam.translate')
        kspaceIn = kspaceIn(:) .* exp( j*(gridParam.translate(1)*currGridX(:)+gridParam.translate(2)*currGridY(:)) );
    end   
    kspaceOut = LibConvGrid( dens1D .* kspaceIn(:), currGridX(:), currGridY(:), gridSize, ones(gridSize,gridSize), gridKern.kernel, gridKern.kernRes);     
end

% 3D gridding
if dim==3
    gridX = traj(:,1);
    gridY = traj(:,2);
    gridZ = traj(:,3);    
   
    gridKern = ConvKernel3( gridSize, gridAlpha, gridW, gridScale ); 
    gridKDeapod = real (LibConvGrid3( 1+j, gridCen, gridCen, gridCen, gridSize, ones(gridSize,gridSize,gridSize), gridKern.kernel, gridKern.kernRes ));
    
    currGridX = gridAlpha*gridX*FOVscale + gridCen;
    currGridY = gridAlpha*gridY*FOVscale + gridCen;
    currGridZ = gridAlpha*gridZ*FOVscale + gridCen;  
    
    [dens1D] = ConvGridDensity3HJ( currGridX(:), currGridY(:), currGridZ(:), gridSize, gridKern, nPar );
    
    if gridParam.nPar > 1
        kspaceOut = LibConvGrid3Par( dens1D .* kspaceIn(:), currGridX(:), currGridY(:), currGridZ(:), gridSize, ...
                ones(gridSize,gridSize,gridSize), gridKern.kernel, gridKern.kernRes, nPar );
    else
        kspaceOut = LibConvGrid3( dens1D .* kspaceIn(:), currGridX(:), currGridY(:), currGridZ(:), gridSize, ...
            ones(gridSize,gridSize,gridSize), gridKern.kernel, gridKern.kernRes );    
    end
end
end
