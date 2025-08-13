function emask=largest_component(image,thresh,varargin)
%
% finds the largest connected component of the thresholded
% image.  returns logical mask for the component.  
% To segment the image (remove everything but this component),
% do the following:
%
% mask=largest_component(image,thresh,varargin);
% image=image .* mask;
%
% can take either a 2D slice or a 3D volume image.
% in the case of a volume image, uses the largest voxel
% value in the volume as the basis for thresholding.
%
% C. Pelizzari August 2005

% can't pass varargin from one function to another as varargin
% you have to expand it as varargin{:}
szin=size(image);
emask={};
if (length(szin) == 2)
% added 1, which gets passed as varargin, new con comp will use it as
% nobjout  CH 01-24-07
    comp=connected_components(image,thresh,1);
    if (numel(comp) > 0)
        emask=comp{1};
    end
    return
elseif (length(szin)==3)
    nsl=szin(3);
    emask=logical(zeros(szin)); %#ok<LOGL>
% ___ connected_components ignores highest 1% of pixels and works on a
% normalized image CP&CH 01-25-07
%     th=thresh*max(image(:));    %use max of entire volume for thresh
% _____
    for i=1:nsl
%         disp(['slice: ', num2str(i)])
        im=squeeze(image(:,:,i));
% ___ connected_components ignores highest 1% of pixels and works on a
% normalized image CP&CH 01-25-07
%         maxim=max(im(:));
%         if (maxim ~= 0 && maxim >= th)  %check if maxim = 0 because th can be zero too  CH 5-25-06
%             comp=connected_components(im,th/maxim,varargin);
% _____
% added 1, which gets passed as varargin, new con comp will use it as
% nobjout  CH 01-24-07
            comp=connected_components(im,thresh,1);
            if (numel(comp) > 0)
%                 disp(size(comp,1))
                emask(:,:,i)=comp{1};
            end
%         end
    end
end
        