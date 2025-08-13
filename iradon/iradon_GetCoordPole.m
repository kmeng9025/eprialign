function [varargout] = iradon_GetCoordPole(pole)

if ischar(pole)
    cpole = pole;
    switch pole
        case 'X', ipole = 1;
        case 'Y', ipole = 2;
        case 'Z', ipole = 3;
        otherwise, error('GetCoordPole: unknown coordinate pole.');
    end
else
    ipole = pole;
    switch pole
        case 1, cpole = 'X';
        case 2, cpole = 'Y';
        case 3, cpole = 'Z';
        otherwise, error('GetCoordPole: unknown coordinate pole.');
    end
end

switch nargout
 case 1,
   varargout = {ipole};
 case 2,
   varargout = {ipole, cpole};
end
