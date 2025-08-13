% angles in Rads
function A = hmatrix_rotate_euler(angles_degree)

A = zeros(4);

angles = angles_degree*pi/180;

% Kang definition
c1 = cos(angles(1)); s1 = sin(angles(1)); 
c2 = cos(angles(2)); s2 = sin(angles(2)); 
c3 = cos(angles(3)); s3 = sin(angles(3)); 

A(1,1)= c2;
A(1,2)= -c3*s2;
A(1,3)= s2*s3;
A(2,1)= c1*s2;
A(2,2)= c1*c2*c3-s1*s3;
A(2,3)= -c3*s1-c1*c2*s3;
A(3,1)= s1*s2;
A(3,2)= c1*s3+c2*c3*s1;
A(3,3)= c1*c3-c2*s1*s3;

% A(1:3,1:3) = erot(Angles * pi/180);
A(4,4)=1;