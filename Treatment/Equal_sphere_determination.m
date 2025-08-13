function [ Inner_radius, Outer_radius ] = Equal_sphere_determination( Radius )


%Takes radius in voxel units. 1 voxel unit *  0.6629  = 1 mm unit.
%Outputs in terms of voxel raidus units.

Inner_radius = Radius

Volume_in_sphere =  (Radius *0.6629)^3 * 4/3 * pi 

Outer_radius =(2*Volume_in_sphere *3/4 * 1/pi )^(1/3)/0.6629

Outer_radius_volume = (Outer_radius *0.6629)^3 * 4/3 * pi 

Shell_volume = Outer_radius_volume-Volume_in_sphere


end

