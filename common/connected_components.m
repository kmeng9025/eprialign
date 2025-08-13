function components=connected_components(image,thresh,varargin)
%
% finds the connected components of the thresholded 2D image
% returns logical masks for each connected component in 
% sorted order, largest component first within a Cell Array
%
% to segment the largest component from the image, do this:
% comp=connected_components(image, thresh);
% then
%   segslice=image;
%   segslice(~comp{1})=0;
% or
%   segslice=image.*comp{1};
%
% if you want just the largest component, use
% comp=connected_components(image, thresh,1);
% C. Pelizzari August 2005
% ignores highest 1% of pixels and works on a
% normalized image CP&CH 01-25-07

% im2bw, bwmorph, bwlabel require 2D images

if (max(image(:)) == 0)
    components={};
    return
end
xhist=0:0.01:1;
ns=hist(image(:)/max(image(:)),xhist);
ins=cumsum(ns(:))/sum(ns(:));
pct99=max(find(ins <= 0.99));
mythresh=xhist(pct99)*thresh;
if isempty(mythresh) || mythresh<0 || mythresh>1, mythresh=thresh; end
mask=im2bw(image/max(image(:)), mythresh);

mask=bwmorph(mask,'open');  % get rid of background noise pixels
mask=bwmorph(mask,'close'); % fill holes in the larger components

[objects,nobj]=bwlabel(mask);  % this labels the components
% nobj=max(objects(:));  % largest label defines number of components
if nobj>=20
    disp([num2str(nobj) ' components found.']);
end
if (nobj > 0)
%     disp('creating cell array');
%     components=cell(nobj,1);
    indices=cell(nobj,1);
    osize=zeros(nobj,1);
    for i=1:nobj
        indices{i}=find(objects==i);  % identify the component
        osize(i)=numel(indices{i}); % find its size
    end
%     disp('sorting');
    [b,ix]=sort(osize,'descend');  % sort them and get index vector
%     disp('filling cell array');
    nobjout=nobj;
    if (nargin > 2)
        nobjout=varargin{1};        
        if iscell(nobjout), nobjout=cell2mat(nobjout); end
        if nobjout>nobj, nobjout=nobj; end  % nobjout can't be greater than nobj                
        components=cell(nobjout,1); % make components after deciding nobj CH 08-29-08
    end
%     fprintf(1,'You get %2.0f objects returned \n', nobjout)
    for i=1:nobjout  % create the output cell array
        mymask=zeros(size(mask));
        mymask(indices{ix(i)})=1;
        components{i}=logical(mymask);
    end
%     disp('Done finding connected comps');
else
    components={};
end