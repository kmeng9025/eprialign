classdef classHalbach
  methods (Static)
    function B = singleMagnet(position, dipoleMoment, simDimensions, resolution)
      x = linspace(-simDimensions(1)/2 + position(1), simDimensions(1)/2 + position(1), floor(simDimensions(1)*resolution+1));
      y = linspace(-simDimensions(2)/2 + position(2), simDimensions(2)/2 + position(2), floor(simDimensions(2)*resolution+1));
      z = linspace(-simDimensions(3)/2 + position(3), simDimensions(3)/2 + position(3), floor(simDimensions(3)*resolution+1));
      [X, Y, Z] = meshgrid(x,y,z);
      
      vec_dot_dip = 3*(X*dipoleMoment(1) + Y*dipoleMoment(2));
      
      %calculate the distance of each mesh point to magnet, optimised for speed
      %for improved memory performance move in to b0 calculations
      vec_mag = X.^2 + Y.^2 + Z.^2;
      vec_mag_3 = vec_mag.^1.5;
      vec_mag_5 = vec_mag.^2.5;
      
      B = zeros(floor(simDimensions(1)*resolution)+1,...
        floor(simDimensions(2)*resolution)+1,...
        floor(simDimensions(3)*resolution)+1,3);
      
      %calculate contributions of magnet to total field, dipole always points in xy plane
      %so second term is zero for the z component
      B(:,:,:,1) = B(:,:,:,1) + X.*vec_dot_dip./vec_mag_5 - dipoleMoment(1) ./ vec_mag_3;
      B(:,:,:,2) = B(:,:,:,2) + Y.*vec_dot_dip./vec_mag_5 - dipoleMoment(2) ./ vec_mag_3;
      B(:,:,:,3) = B(:,:,:,3) + Z.*vec_dot_dip./vec_mag_5;
    end
%     function H = create(numMagnets = 24, rings = (-0.075,-0.025, 0.025, 0.075), radius = 0.145, magnetSize = 0.0254, kValue = 2, resolution = 1000, bRem = 1.3, simDimensions = (0.3, 0.3, 0.2))
%       % define vacuum permeability
%       mu = 1e-7;
%       
%       %positioning of the magnets in a circle
%       angle_elements = linspace(0, 2*pi, numMagnets);
%       
%       %Use the analytical expression for the z component of a cube magnet to estimate
%       %dipole momentstrength for correct scaling. Dipole approximation only valid
%       %far-ish away from magnet, comparison made at 1 meter distance.
%       
%       dip_mom = magnetization(bRem, magnetSize)
%       
%       %create array to store field data
%       B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3), dtype=np.float32)
%       
%       %create halbach array
%       for row in rings:
%         for angle in angle_elements:
%           position = (radius*np.cos(angle),radius*np.sin(angle), row)
%           
%           dip_vec = [dip_mom*np.cos(kValue*angle), dip_mom*np.sin(kValue*angle)]
%           dip_vec = np.multiply(dip_vec,mu)
%           
%           %calculate contributions of magnet to total field, dipole always points in xy plane
%           %so second term is zero for the z component
%           B0 += singleMagnet(position, dip_vec, simDimensions, resolution)
%           
%           %           return B0
%         end
%       end
%     end
  end
end