function sp1=rotateXI(sp,fi);

a=fi*pi/180;
complex=fromImToComplex(sp);  
sp1=imag( complex* exp(i*a) ); % rotated spectrum
sp1=zeroLine(sp1,.05);
