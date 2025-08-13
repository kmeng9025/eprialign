function [ Sphere_mask ] = Create_Sphere_mask( Isocenter, Image, Vox_radius )

%This fuction creates a spherical mask centered at Isocenter [x,y,z] with
%Vox radius included.

Sphere_mask = zeros(size(Image));
Dist_mat = zeros(size(Image))
x = Isocenter(1)
y = Isocenter(2)
z = Isocenter(3)


for ii = 1:size(Image,1);
    for jj = 1:size(Image,2);
        for kk = 1:size(Image,3);
            Dist_mat(ii,jj,kk) = ((ii-x)^2+(jj-y)^2+(kk-z)^2)^(1/2);
        end
    end
end

Sphere_mask(find(Dist_mat <= Vox_radius)) = 1; 



end