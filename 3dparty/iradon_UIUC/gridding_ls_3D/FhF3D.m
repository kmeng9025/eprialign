function out = FhF3D(image,Qf)
bigIm = zeros(size(Qf));
N1 = size(Qf,1)/2;
N2 = size(Qf,2)/2;
N3 = size(Qf,3)/2;
[n2, n1, n3] = meshgrid(N2+1+[-N2/2:N2/2-1],N1+1+[-N1/2:N1/2-1],N3+1+[-N3/2:N3/2-1]);
ind = sub2ind(2*[N1,N2,N3],n1(:),n2(:),n3(:));
bigIm(ind) = image(:);

bigIm2 = fftshift(ifftn(fftn(ifftshift(bigIm)).*Qf));
out = bigIm2(ind);
