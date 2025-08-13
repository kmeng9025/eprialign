%RADON_ANGLE2XYZ   Conversion of angles into unit vector projections on
%coordinate axis
% radon_pars = RADON_ANGLE2XYZ(radon_pars)
% radon_radon_pars - structure of parameters
%   x,y,z arrays of unit vector projections on coordinate axis
%   Theta, Phi - projection angles
%
%   See also RADON_D2D.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function radon_pars = radon_angle2xyz(radon_pars)

Theta = radon_pars.Theta(:);
Phi = radon_pars.Phi(:); 
radon_pars.x = sin(Theta).*cos(Phi);
radon_pars.y = sin(Theta).*sin(Phi);
radon_pars.z = cos(Theta);