% angles in Rads
function A = hmatrix_rotate_euler(Angles)

A = zeros(4);

% Kang definition
psi = Angles(1); phi = Angles(2); theta = Angles(3);
A(1,1)= cos(psi)*cos(phi)-cos(theta)*sin(phi)*sin(psi);
A(1,2)= cos(psi)*sin(phi)+cos(theta)*cos(phi)*sin(psi);
A(1,3)= sin(psi)*sin(theta);
A(2,1)= -sin(psi)*cos(phi)-cos(theta)*sin(phi)*cos(psi);
A(2,2)= -sin(psi)*sin(phi)+cos(theta)*cos(phi)*cos(psi);
A(2,3)= cos(psi)*sin(theta);
A(3,1)= sin(theta)*sin(phi);
A(3,2)= -sin(theta)*cos(phi);
A(3,3)= cos(theta);

% A(1:3,1:3) = erot(Angles * pi/180);
A(4,4)=1;