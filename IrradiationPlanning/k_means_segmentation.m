function [ SKIN, PVS, ring, bg ] = k_means_segmentation( CT )
%CT Segmantation based on k0 means matlab algoritim 

CT = CT;

CT2D = CT(:,:,175);


[coords_bg] = [12,12];
[coords_ring] = [112,78];
% [xpvs,ypvs]=ginput(1);
% [xskin,yskin] = ginput(1);

% ind_bg = sub2ind(size(CT2D), 12, 12);
% ind_ring = sub2ind(size(CT2D), 78, 112);
% ind_pvs = sub2ind(size(CT2D), ypvs, xpvs);
% ind_skin = sub2ind(size(CT2D), yskin, xskin);
% ind_pvs = sub2ind(size(CT2D), 161, 109);
% ind_skin = sub2ind(size(CT2D), 187, 196);

% coords = round([ind_bg,ind_ring,ind_pvs,ind_skin])';

seeds =1.0e+03 *[-0.8219,1.1711];
seeds = seeds';

  
    
[output] = kmeans(CT2D(:),2,'Start',seeds);

output1 = reshape(output,[size(CT,1),size(CT,2),1]);
imagesc(output1)
%out = reshape(z,[size(CT,1),size(CT,2),size(CT,3)]);

bg_vec = output==1;
bg = reshape(bg_vec,[size(CT2D,1),size(CT2D,2)]);

all_vec = output==2;
all = reshape(all_vec,[size(CT2D,1),size(CT2D,2)]);

seedsnew = 1.0e+03 *[0.7895;3.3958;0.0075;4.9841];
new = CT2D.*all;
new1 = kmeans(new(:),4,'Start',seedsnew);
new1 = reshape(new1,[size(CT,1),size(CT,2),1]);
imagesc(new1);


pvs_vec = new1==2;
pvs = reshape(pvs_vec,[size(CT2D,1),size(CT2D,2)]);


ring_skin_vec = new1==1;
ring_skin = reshape(ring_skin_vec,[size(CT2D,1),size(CT2D,2)]);



new2 = CT2D.*ring_skin;

seeds3=1.0e+03 * [0;1.1334;0.6979];
new3 = kmeans(new2(:),3,'Start',seeds3);
new3 = reshape(new3,[size(CT,1),size(CT,2),1]);

skin_vec = new3==2;
skin1 = reshape(skin_vec,[size(CT2D,1),size(CT2D,2)]);
skin2 = bwmorph(skin1,'majority');
skin3 = bwmorph(skin2,'majority');
skin4 = bwmorph(skin3,'majority');
skin5 = bwmorph(skin4,'majority');
skin6 = bwmorph(skin5,'majority');
skin7 = bwmorph(skin6,'majority');
skin8 = bwmorph(skin7,'clean');
skin9= imfill(skin8,'holes');
skin10 = bwareaopen(skin9,1000);
SKIN = skin10;

PVS = pvs-and(SKIN,pvs);


ring = xor(all,SKIN);
ring = xor(ring,PVS);

ring1 = bwmorph(ring,'majority');


cc = regionprops(ring1,'Area');
maxarea = max([cc.Area]);
out = bwareaopen(ring1,maxarea);


subplot(2,2,1);
imagesc(bg);
subplot(2,2,2);
imagesc(SKIN);
subplot(2,2,3);
imagesc(PVS);
subplot(2,2,4);
imagesc(out);





k=1;
end

