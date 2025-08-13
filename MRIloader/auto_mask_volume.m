function maskvol = auto_mask_volume(im)

maskvol = zeros(size(im));
% globalmax = max(im(:));

nslices = size(im,3);

for n = 1:nslices
    myslice = squeeze(im(:,:,n));
    [npix,xout] = hist(myslice(:), 100);
    posderivs = find(diff(npix) > 0);
%     npeak = find(npix == max(npix(:)));
    npeak = find(npix == max(npix(:)),1,'last'); % only need one CH 3-17-08
    nfew = find(npix < 0.1 * max(npix));  % histogram bins that 
                                              % are not very full
    nfewright = nfew(find(nfew > npeak, 1 )); % well down the peak
    
    threshold1 = xout(posderivs(find(posderivs > nfewright, 1 )));
    mymask = myslice;
    % return if threshold is empty (blank slice) CH 07-12-08
    if isempty(threshold1), return, end
    mymask(mymask < threshold1) = 0;
    mymask(mymask > 0) = 1;
    %[l,num] = bwlabel(mymask);
    %pixels = zeros(num,1);
  
    %for nl=1:num
    %    pixels(nl) = prod(size(find(l == nl)));
    %end
    
    %nlmax = find(pixels == max(pixels(:)));
    %for ln = 1:size(nlmax,1)
    %    mymask(find(l ~= nlmax(ln))) = 0;
    %end % retain only the largest component(s)
    
    mymask = imfill(mymask,'holes');
    mymask = bwmorph(mymask,'erode');
    mymask = bwmorph(mymask,'dilate');

    maskvol(:,:,n) = mymask;
end
        
