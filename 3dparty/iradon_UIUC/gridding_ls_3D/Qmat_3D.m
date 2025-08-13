function Qf = Qmat_3D(N1, N2, N3, kx, ky, kz, num_sam)

beta = 18.5547;
gridOS = 2;
kernelWidth=4;
[cartesian_grid] = reshape(grid3d(2*kx, 2*ky, 2*kz, 1/(N1*N2*N3)*ones(num_sam, 1), 2*N1, 2*N2, 2*N3, gridOS, kernelWidth, beta), gridOS*N1*2, gridOS*N2*2, gridOS*N3*2);
for ii = 1:4
    Q2(:, :, (ii-1)*N3+1:ii*N3)  = fftshift(ifft2(ifftshift(cartesian_grid(:, :, 1:N3))));
    cartesian_grid = cartesian_grid(:, :, N3+1:end);
end
clear cartesian_grid
for ii = 1:4
    Q2((ii-1)*N1+1:ii*N1, :, :) = reshape(fftshift(ifft(ifftshift(reshape(Q2((ii-1)*N1+1:ii*N1, :, :), N1*gridOS*N2*2, gridOS*N3*2), 2), [], 2), 2), N1, gridOS*N2*2, gridOS*N3*2);
end
Q2 = prod([4*N1, 4*N2, 4*N3])*Q2;
[kernelX kernelY kernelZ] = meshgrid([-N2:N2-1]/N2/2, [-N1:N1-1]/N1/2, [-N3:N3-1]/N3/2);
    gridKernel = (sin(sqrt(pi^2*kernelWidth^2*kernelX.^2 - beta^2))./ ...
        sqrt(pi^2*kernelWidth^2*kernelX.^2 - beta^2)).*(sin ...
        (sqrt(pi^2*kernelWidth^2*kernelY.^2 -beta^2))./sqrt(pi^2*kernelWidth...
        ^2*kernelY.^2 - beta^2)).*(sin ...
        (sqrt(pi^2*kernelWidth^2*kernelZ.^2 -beta^2))./sqrt(pi^2*kernelWidth...
        ^2*kernelZ.^2 - beta^2));    

clear kernelX kernelY kernelZ
Q = Q2(2*N1+1+[-N1:N1-1], 2*N2+1+[-N2:N2-1], 2*N3+1+[-N3:N3-1])./gridKernel;
clear Q2 gridKernel
Qf = fftn(fftshift(Q));
