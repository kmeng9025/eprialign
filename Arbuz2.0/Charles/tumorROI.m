function tumorPts=tumorROI(mri)
%tumorPts=tumorROI(m38cast2_6sag)

nslice=size(mri,3);
tumorPts=cell(2,nslice);
for thisslice=1:nslice
imagesc(squeeze(mri(:,:,thisslice)),...
    [max(mri(:))*0.01 max(mri(:))*0.95]);
axis square;
[x,y,BW,xi,yi] = roipoly;

    if (size(xi,1)>5)
    % use cell array
    tumorPts(1,thisslice)={xi};
    tumorPts(2,thisslice)={yi};
    end 
end %thisslice to nslice
clear mri thisslice nslice x y BW xi yi
% save ROI to file
[filename pathname] = uiputfile('*.txt','File to save Tumor ROI pts?');

if (filename ~= 0)
write_3d_contours(tumorPts,filename)
end