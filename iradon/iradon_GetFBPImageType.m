% IRADON_GETFBPIMAGETYPE returns numerical and string representation of
% image type
% [nimtype, simtype] = iradon_GetFBPImageType(imtype);
% imtype - Image type [int, 1 to 14]
%            or
%        - Image type [XB/YB/ZB/XYB/XZB/YZB/XYZB/ X/Y/Z/XY/XZ/YZ/XYZ]
% nimage  - Image type [int, 1 to 14]
% simtype - Image type [XB/YB/ZB/XYB/XZB/YZB/XYZB/ X/Y/Z/XY/XZ/YZ/XYZ]
% 
% See also IRADON_FBP_GRAD_TABLE

function [varargout] = iradon_GetFBPImageType(imtype)

if ischar(imtype)
    ctype = imtype;
    switch imtype
        case 'XYZB', itype = 1;
        case 'XZB', itype = 2;
        case 'YZB', itype = 3;
        case 'XYB', itype = 4;
        case 'XB', itype = 5;
        case 'YB', itype = 6;
        case 'ZB', itype = 7;
        case 'X', itype = 8;
        case 'Y', itype = 9;
        case 'Z', itype = 10;
        case 'XZ', itype = 11;
        case 'YZ', itype = 12;
        case 'XY', itype = 13;
        case 'XYZ', itype = 14;
        otherwise, error('GetImType: unknown image type.');
    end
else
    itype = imtype;
    switch imtype
        case 1, ctype = 'XYZB';
        case 2, ctype = 'XZB';
        case 3, ctype = 'YZB';
        case 4, ctype = 'XYB';
        case 5, ctype = 'XB';
        case 6, ctype = 'YB';
        case 7, ctype = 'ZB';
        case 8, ctype = 'X';
        case 9, ctype = 'Y';
        case 10, ctype = 'Z';
        case 11, ctype = 'XZ';
        case 12, ctype = 'YZ';
        case 13, ctype = 'XY';
        case 14, ctype = 'XYZ';
        otherwise, error('GetImType: unknown image type.');
    end
end

switch nargout
 case 1,
   varargout = {itype};
 case 2,
   varargout = {itype, ctype};
end
