function Fhd = Fuh_nonCart_3D(d, kx, ky, kz, N1, N2, N3)

beta = 18.5547;
gridOS = 2;
kernelWidth=4;
[cartesian_grid] = reshape(grid3d(kx, ky, kz, d, N1, N2, N3, gridOS, kernelWidth,beta), gridOS*N1, gridOS*N2, gridOS*N3);
Q2 = sqrt(prod([2*N1, 2*N2, 2*N3]))*fftshift(ifftn(ifftshift(cartesian_grid)));
clear cartesian_grid
[kernelX kernelY kernelZ] =meshgrid([-N2/2:N2/2-1]/N2, [-N1/2:N1/2-1]/N1, [-N3/2:N3/2-1]/N3);
gridKernel = (sin(sqrt(pi^2*kernelWidth^2*kernelX.^2 - beta^2))./ ...
        sqrt(pi^2*kernelWidth^2*kernelX.^2 - beta^2)).*(sin ...
        (sqrt(pi^2*kernelWidth^2*kernelY.^2 -beta^2))./sqrt(pi^2*kernelWidth...
        ^2*kernelY.^2 - beta^2)).*(sin ...
        (sqrt(pi^2*kernelWidth^2*kernelZ.^2 -beta^2))./sqrt(pi^2*kernelWidth...
        ^2*kernelZ.^2 - beta^2));    
clear kernelX kernelY kernelZ
Fhd = Q2(N1+1+[-N1/2:N1/2-1], N2+1+[-N2/2:N2/2-1], N3+1+[-N3/2:N3/2-1])./gridKernel;
Fhd = Fhd(:);

