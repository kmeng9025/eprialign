

doimage=1;
doplot=1;
figure; 
subplot(1,3,1), imagesc(midslice),axis image, colormap gray;
subplot(1,3,3); imax=gca;
subplot(1,3,2);  plotax=gca;
inring=(rsqmesh1 < rin*rin);
if min(midslice(:))<0, midslice=midslice-min(midslice(:));end
[nr,cr]=hist(midslice(inring),200);

if doplot, 
    axes(plotax),plot(cr,nr,'.');
    hold on
end
big=find(nr>=0.3*max(nr));
bigrange=cr(big(end))-cr(big(1));
firstpeak=min(big);  %this will be the air peak - actually a tall peak
                           % with a shorter one right next to it
%first, the local maximum                           
localpeak=max(nr(1:firstpeak+5));
ipeak=min(find(nr(1:firstpeak+5)==localpeak));
%next, a little to its right
ishoulder=ipeak-1+min(find(nr(ipeak:end)<=0.3*localpeak));
par1=[cr(ipeak) localpeak cr(ipeak+1)-cr(ipeak-1);...
      cr(ishoulder) nr(ishoulder) cr(ishoulder+1)-cr(ishoulder-1)];
options=optimset('MaxFunEval',50000,'MaxIter',20000);
myfunc=@(x)gauss_peaks_deviation(x,cr,nr);
[xpar1,chisq]=fminsearch(myfunc,par1,options);
hipos=max(xpar1(:,1));
lopos=min(xpar1(:,1));
% check if one peak wandered over to the tissue cluster.  reject if so
if (hipos-lopos) > 0.25*bigrange,
    allpars=xpar1(find(xpar1(:,1)==lopos),:);
else
    allpars=xpar1;
end
lopeak=gauss_peaks_value(allpars,cr);
resid=nr-lopeak;
if doplot, 
    %plot(cr,resid,'.'); 
    plot(cr,lopeak);
end

lomarg=cumsum(lopeak)/sum(lopeak);
cutoff=cr(min(find(lomarg>0.999)));
airmask=inring&midslice<=cutoff;
% now a peak at the high end for the PVS

hiedge=max(find(resid>=0.3*max(resid)));
localpeak=max(resid(hiedge-5:end));
peakpos=cr(min(find(resid(hiedge-5:end)==localpeak))+hiedge-6);
baseline=hiedge+min(find(resid(hiedge:end)<0.1*localpeak) );
leftend=max(find(resid(1:hiedge)<0.1*localpeak));
par2=[peakpos localpeak (cr(baseline)-peakpos)/4;...
      peakpos-(cr(baseline)-peakpos)/4 localpeak/4 (cr(baseline)-peakpos) ];

fake=resid;
fake(1:leftend-1)=0;
myfunc=@(x)gauss_peaks_deviation(x,cr,fake);
[xpar2,chisq]=fminsearch(myfunc,par2,options);

candidates=find(xpar2(:,1) > cr(leftend));
pvspeak=find(xpar2(candidates,2)==max(xpar2(candidates,2)));
ppeak=gauss_peaks_value(xpar2,cr);
pvsmar=cumsum(ppeak)/sum(ppeak);

pvspeak=pvspeak + size(allpars,1);
allpars=[allpars;xpar2];
nuresid=nr-gauss_peaks_value(allpars,cr);
if doplot, 
    %plot(cr,nuresid,'.'); 
    plot(cr,ppeak);
end

% find the max of residual in the range (0.3,0.6) of the interval between
% the air peak and the pvs peak - this should be the tissue peak
losearch=allpars(1,1); % the low peak
hisearch=allpars(pvspeak,1); % the high peak

srange=losearch+(hisearch-losearch)*[0.3 0.6];
svals=find(cr>=srange(1)&cr<=srange(2));
thepeak=find(nuresid==max(nuresid(svals)));
fullrange=find(cr>=losearch&cr<=hisearch);
fullave=(sum(nuresid(fullrange)) - sum(nuresid(svals)))/(numel(fullrange)-numel(svals));
peak_to_back=mean(nuresid(svals))/fullave
% a peak and a broad background for the leg
par4=[cr(thepeak(1)) nuresid(thepeak(1))-fullave allpars(pvspeak,3)];%...
    %0.5*(allpars(pvspeak,1)+allpars(1,1)) nuresid(thepeak(1))/10 0.25*(allpars(pvspeak,1)-allpars(1,1))];

myfunc=@(x)gauss_peaks_deviation(x,cr(svals),nuresid(svals)-fullave);
[xpar5,chisq]=fminsearch(myfunc,par4,options);
allpars=[allpars;xpar5];
nuresid2=nr-gauss_peaks_value(allpars,cr);
if doplot,
    plot(cr,nuresid2,'.');
    plot(cr,tispeak+fullave);
    pause(0.01)
end
goodpeak=find(xpar5(:,1)>=srange(1) & xpar5(:,1)<=srange(2));
if numel(goodpeak)>1,  % use the narrower one
    igood=find(xpar5(goodpeak,3)==min(xpar5(goodpeak,3)));
    goodpeak=goodpeak(igood);
end
pvslo=cr(min(find(pvsmar>0.01)));
tispeak=gauss_peaks_value(xpar5(goodpeak,:),cr);
tismar=cumsum(tispeak)/sum(tispeak);
tislo=cr(min(find(tismar>0.001)));
tislo=tislo-0.25*(tislo-cutoff);
tishi=cr(min(find(tismar>0.99)));
tishi=tishi+0.25*(pvslo-tishi);

pvs=inring&~airmask&midslice>tishi;

% get rid of pixels on the pvs-air interface
pvair=imdilate(pvs,strel('disk',7))+imdilate(airmask,strel('disk',7));

tis=inring&~airmask&~pvs&pvair<2&midslice>=tislo&midslice<=tishi;
tisopen=bwmorph(tis,'open');
% find the biggest component, which is the leg
CC = bwconncomp(tisopen);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
tismask=0*tis;
tismask(CC.PixelIdxList{idx})=1;
tismask=imfill(tismask,'holes');
% some high density voxels inside tissue (bone?) might have been misclassified
hotspots=tismask&midslice>tishi;
tismask(hotspots)=0;

%tismask=bwmorph(tismask,'open');
%pvsmask=inring&~tismask&~airmask&midslice>tishi;
pvs=inring&~airmask&midslice>0.5*(tislo+tishi);
pvsmask=pvs&~tismask;
totmask=airmask+2*tismask+3*pvsmask;
if doimage,
    axes(imax),
    imagesc(totmask),axis image,colormap(gca,jet_black);
    pause(0.01);
end

