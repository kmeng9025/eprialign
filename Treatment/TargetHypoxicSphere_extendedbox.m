% Similar to TargetHypoxicSphere function but now finds smallest 3D box
% containing tumor mask and extends this box by radius and searches each
% voxel within the box as possible isocenters.

function [isocenter,results] = TargetHypoxicSphere_extended(po2image,radius,Mask)

% choice = questdlg('Which mask has been supplied?','Mask Type','Tumor Mask','EPR Mask','Cancel','Tumor Mask');
% switch choice
%     case 'Tumor Mask'
%         
%         tic
        
        tumormask = Mask;
        
        tumorvox = numel(find(tumormask == 1));
        intumor = po2image(tumormask == 1);
        hypoxicvox = numel(find(intumor <= 10));
        normoxvox = numel(find(intumor > 10));
        
        center = zeros(tumorvox,3);
        hypoxic_tumor_insphere = [];
        normoxic_tumor_insphere =  [];
        normtiss_insphere = [];
        tumor_outsphere = [];
        
        %find box containing tumor
aa = 0;
for ii = 1:size(po2image,1)
    plane = squeeze(tumormask(ii,:,:));
    test = numel(find(plane == 1));
    if aa == 0 && test > 0
        aa = 1;
        minx = ii;
    elseif aa == 1 && test == 0
        aa = 0;
        maxx = ii-1;
    end
end
aa = 0;
for ii = 1:size(po2image,2)
    plane = squeeze(tumormask(:,ii,:));
    test = numel(find(plane == 1));
    if aa == 0 && test > 0
        aa = 1;
        miny = ii;
    elseif aa == 1 && test == 0
        aa = 0;
        maxy = ii-1;
    end
end
aa = 0;
for ii = 1:size(po2image,3)
    plane = squeeze(tumormask(:,:,ii));
    test = numel(find(plane == 1));
    if aa == 0 && test > 0
        aa = 1;
        minz = ii;
    elseif aa == 1 && test == 0
        aa = 0;
        maxz = ii-1;
    end
end
box = zeros(64,64,64);
box(minx-radius:maxx+radius,miny-radius:maxx+radius,minz-radius:maxz+radius) = 1;
        
        count1 = 0;
        for ii = 1:size(po2image,1)
            for jj = 1:size(po2image,2)
                for kk = 1:size(po2image,3)
                    if box(ii,jj,kk) == 1
                        count1 = count1 + 1;
                        count2 = 0; count3 = 0; count4 = 0; count5 = 0;
                        center(count1,:) = [ii, jj, kk];
                        for mm = 1:size(po2image,1)
                            for nn = 1:size(po2image,2)
                                for pp = 1:size(po2image,3)
                                    dist = sqrt((ii-mm)^2 + (jj-nn)^2 + (kk-pp)^2);
                                    if tumormask(mm,nn,pp) == 1 && dist <= radius && po2image(mm,nn,pp) <= 10
                                        count2 = count2 + 1;
                                    elseif tumormask(mm,nn,pp) == 0 && dist <= radius
                                        count3 = count3 + 1;
                                    elseif tumormask(mm,nn,pp) == 1 && dist <= radius && po2image(mm,nn,pp) > 10
                                        count4 = count4 + 1;
                                    elseif tumormask(mm,nn,pp) == 1 && dist > radius
                                        count5 = count5 + 1;
                                    end
                                end
                            end
                        end
                        hypoxic_tumor_insphere(count1) = count2/hypoxicvox;
                        normtiss_insphere(count1) = count3;
                        normoxic_tumor_insphere(count1) = count4/normoxvox;
                        tumor_outsphere(count1) = count5/tumorvox;
                    end
                end
            end
        end
        [~,maxidx] = max(hypoxic_tumor_insphere);
        if length(maxidx) > 1
            temp = normtiss_insphere(maxidx);
            [~,minidx] = min(temp);
            bestidx = maxidx(minidx);
        else
            bestidx = maxidx;
        end
        results.hypoxic_tumor_insphere_fraction = hypoxic_tumor_insphere(bestidx);
        results.hypoxic_tumor_insphere_volume = hypoxic_tumor_insphere(bestidx)*hypoxicvox;
        results.normtiss_insphere = normtiss_insphere(bestidx);
        results.normoxic_tumor_insphere_fraction = normoxic_tumor_insphere(bestidx);
        results.normoxic_tumor_insphere_volume = normoxic_tumor_insphere(bestidx)*normoxvox;
        results.tumor_outsphere = tumor_outsphere(bestidx);
        
        isocenter = center(bestidx,:);
        
%         toc
        
%     case 'EPR Mask'
%         
%         tic
%         
%         eprmask = Mask;
%         
%         eprvox = numel(find(eprmask == 1));
%         inepr = po2image(eprmask == 1);
%         hypoxicvox = numel(find(inepr <= 10));
%         normoxvox = numel(find(inepr > 10));
%         
%         center = zeros(eprvox,3);
%         
%         hypoxic_insphere = [];
%         normoxic_insphere = [];
%         
%         count1 = 0;
%         for ii = 1:size(po2image,1)
%             for jj = 1:size(po2image,2)
%                 for kk = 1:size(po2image,3)
%                     if eprmask(ii,jj,kk) == 1
%                         count1 = count1 + 1;
%                         count2 = 0; count3 = 0; count4 = 0; count5 = 0;
%                         center(count1,:) = [ii, jj, kk];
%                         for mm = 1:size(po2image,1)
%                             for nn = 1:size(po2image,2)
%                                 for pp = 1:size(po2image,3)
%                                     dist = sqrt((ii-mm)^2 + (jj-nn)^2 + (kk-pp)^2);
%                                     if eprmask(mm,nn,pp) == 1 && dist <= radius && po2image(mm,nn,pp) <= 10
%                                         count2 = count2 + 1;
%                                     elseif eprmask(mm,nn,pp) == 1 && dist <= radius && po2image(mm,nn,pp) > 10
%                                         count3 = count3 + 1;
%                                     end
%                                 end
%                             end
%                         end
%                         hypoxic_insphere(count1) = count2/hypoxicvox;
%                         normoxic_insphere(count1) = count3/normoxvox;
%                     end
%                 end
%             end
%         end
%         [~,maxidx] = max(hypoxic_insphere);
%         if length(maxidx) > 1
%             temp = normoxic_insphere(maxidx);
%             [~,minidx] = min(temp);
%             bestidx = maxidx(minidx);
%         else
%             bestidx = maxidx;
%         end
%         results.hypoxic_insphere_fraction = hypoxic_insphere(bestidx);
%         results.hypoxic_insphere_volume = hypoxic_insphere(bestidx)*hypoxicvox;
%         results.normoxic_insphere_fraction = normoxic_insphere(bestidx);
%         results.normoxic_insphere_volume = normoxic_insphere(bestidx)*normoxvox;
%         
%         isocenter = center(bestidx,:);
%         
%         toc
%         
% end