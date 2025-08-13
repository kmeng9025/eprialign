function [skinmask, ringmask, varargout]=findskin(ctvals, varargin)
% FINDSKIN - finds mouse leg surface, outer cylindrical holder, PVS leg
% cushion.
% input - ctvals:  3D array of CT attenuation coefficients 
%   (optional) suppress_artifacts: 1 (default) to suppress regions with
%   metal shields, 0 not to suppress.
% output - skinmask: mask of leg
%           ringmask: mask of outer cylinder
%   (optional) pvmask:  mask of PVS leg immobilization cushion
% 
% C. Pelizzari June 2016


skinmask=[];
ringmask=[];
if nargout>2, varargout{1}=[]; end

if max(ctvals(:)) > 20,
    fprintf('CT numbers don''t look like attenuation coefficients!\n')
    fprintf('Maybe you need to do a transformation like (CT+1000)/5000?\n');
    return
end
cubecopy=ctvals;
suppress_artifacts=1;
if(nargin > 1) suppress_artifacts=varargin{1}; end;

if (suppress_artifacts)
    disp('suppressing scan regions with metal artifacts');
    % identify voxels with metal
    contaminated=find(cubecopy>4);
    [ic,jc,kc]=ind2sub(size(cubecopy),contaminated);
    % find slices without metal
    [nh,ch]=hist(kc,200);
    kgood=ch(find(nh==0));
    kcutlo=round(min(kgood)+0.5);
    kcuthi=round(max(kgood)-0.5);

    cubecopy(:,:,1:kcutlo)=0;
    cubecopy(:,:,kcuthi:end)=0;
end



disp('finding leg mask');
lmask=zeros(size(cubecopy));
lolim=0.36; 
uplim=0.45;
lmask(find(cubecopy>lolim&cubecopy<uplim))=1;
lmask2=imopen(lmask,strel('square',3));
cc=bwconncomp(lmask2);
numpix=cellfun(@(x)numel(x),cc.PixelIdxList);
biggest=find(numpix==max(numpix));
smask=zeros(size(cubecopy));
smask(cc.PixelIdxList{biggest})=1;
% this is the skin
skinmask=imfill(smask,'holes');

disp('finding ring mask');
rmask=zeros(size(cubecopy));
lolim=0.05; 
uplim=0.36;
rmask(find(cubecopy>lolim&cubecopy<uplim))=1;
rmask2=imopen(rmask,strel('square',3));
cc=bwconncomp(rmask2);
numpix=cellfun(@(x)numel(x),cc.PixelIdxList);
biggest=find(numpix==max(numpix));
rmask=zeros(size(cubecopy));
rmask(cc.PixelIdxList{biggest})=1;
% this is the outer ring surrounding the cushion
ringmask=imfill(rmask,'holes');

if nargout >2,
    disp('finding PVS mask');
    pvmask=zeros(size(cubecopy));
    lolim=0.8; 
    uplim=1.03;
    pvmask(find(cubecopy>lolim&cubecopy<uplim))=1;
    pvmask2=imopen(pvmask,strel('square',3));
    cc=bwconncomp(pvmask2);
    numpix=cellfun(@(x)numel(x),cc.PixelIdxList);
    biggest=find(numpix==max(numpix));
    pvmask=zeros(size(cubecopy));
    pvmask(cc.PixelIdxList{biggest})=1;
    % this is the PVSO cushion
    pvmask2=imfill(pvmask,'holes');
    varargout{1}=pvmask2;
end

