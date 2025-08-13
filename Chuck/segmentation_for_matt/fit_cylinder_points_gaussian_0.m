
%CT_data = padarray(CT_data,[100,100,100]);
nsamples=5;
separation=20;
legvol=zeros(size(CT_data));
ringvol=legvol;
pvsvol=legvol;

cubecopy=double(CT_data);
suppress_artifacts=1;
kcutlo=1;
kcuthi=size(cubecopy,3);

if (suppress_artifacts)
    disp('suppressing scan regions with metal artifacts');
    % identify voxels with metal
    
    if max(cubecopy(:)) > 1000, artifact=8000, else artifact = 4, end
    contaminated=find(cubecopy>artifact);
    [ic,jc,kc]=ind2sub(size(cubecopy),contaminated);
    % find slices without metal
    [nh,ch]=hist(kc,200);
    kgood=ch(find(nh==0));
    figure,plot(ch,nh,'.');pause(0.01);
    kcutlo=round(min(kgood)+0.5);
    kcuthi=round(max(kgood)-0.5);

    cubecopy(:,:,1:kcutlo)=0;
    cubecopy(:,:,kcuthi:end)=0;
end
minval=min(cubecopy(:));
maxval=max(cubecopy(:));
%cubecopy=(cubecopy-minval)/(maxval-minval);
ncols=10;
imfig=figure;
nslices=ceil(kcuthi-kcutlo+1)/5;
nrows=ceil(nslices/ncols);
j=1;
for n=kcutlo:5:kcuthi
    figure(imfig);

    subplot(nrows,ncols,j);
    j=j+1;
    imagesc(cubecopy(:,:,n)),axis image,colormap gray
    set(gca,'XTick',[],'YTick',[])
    title(num2str(n));
    pause(0.05);
end

answers=inputdlg({'Starting slice', 'Ending Slice'});
startslice=str2num(answers{1});
endslice=str2num(answers{2});

legval=NaN;
plotfig=figure;
rleg=0;
cleg=0;
for midplane=startslice:endslice
inner=[];
outer=[];
%midplane=round(size(cubecopy,3)/2) -30;
midslice=cubecopy(:,:,midplane);
wimage=gradientweight(midslice);
%gradmask=wimage>0.43;
gradmask=im2bw(wimage,graythresh(wimage));
gradmask=imclose(gradmask,strel('disk',2)); %fill small holes - streaks?
gradmask(:,1)=1;
gradmask(1,:)=1;
gradmask(:,size(gradmask,2))=1;
gradmask(size(gradmask,1),:)=1;
horzproj=sum(~gradmask,2);
tabletop=min(find(horzproj>=size(midslice,2)/2))-2;
ringtop=min(find(horzproj>20));
if tabletop < numel(horzproj), gradmask(tabletop:size(gradmask,1),:)=1; end% erase table
if ringtop > 2, gradmask(1:ringtop-1,:)=1; end  % erase hot spots above ring
vertproj=sum(~gradmask,1);
ringleft=min(find(vertproj>20));
ringright=max(find(vertproj>20));
if ringleft > 3, gradmask(:,1:ringleft-2)=1; end
if ringright < numel(vertproj)-2, gradmask(:,ringright+2:end)=1;end

gradmask=imclose(gradmask,strel('disk',2)); %fill small holes - streaks?

gradmask(:,1)=1;
gradmask(1,:)=1;
gradmask(:,size(gradmask,2))=1;
gradmask(size(gradmask,1),:)=1;
midcol=round(size(midslice,2)/2);

locol=midcol-separation*floor(nsamples/2);
hicol=midcol+separation*floor(nsamples/2);
profcols=locol:separation:hicol;
figure (plotfig);subplot(1,4,1);imagesc(midslice),axis image,colormap gray
title(sprintf('slice %d',midplane));
hold on
subplot(1,4,2);imagesc(gradmask),axis image,colormap gray
hold on
rimthick=0;
topleft=midslice(1:10,1:10);
airval=mean(topleft(:));
for ncol=1:numel(profcols);
    thiscol=profcols(ncol);

    vertprof=midslice(:,thiscol);
    gradprof=gradmask(:,thiscol);
    
    npts=numel(vertprof);
%     boxlen=6;
%     running=[];
%     moving=[];
%     for n=1:npts-boxlen
%         running(end+1)=sum(vertprof(n:n+boxlen-1))/boxlen;
%     end
%     for n=1:boxlen:npts-boxlen
%         moving(end+1)=sum(vertprof(n:n+boxlen-1))/boxlen;
%     end
% 
%     mdiff=diff(moving);
%     rises=find(mdiff>500);
%     falls=find(mdiff<-500);
% 
%     upedge=min(rises);
%     downedge=min(falls);
% 
%     uppos=boxlen*(upedge+0.5);
%     downpos=boxlen*(downedge-0.5);
%     
%     plateau=mean(vertprof(uppos:downpos));
%     initial=mean(vertprof(1:uppos-3*boxlen));
%     
%     halfheight=0.5*(plateau+initial);
%     halfup=min(find(vertprof>=halfheight));
%     halfdown=min(find(vertprof(downpos:end)<=halfheight))+downpos-1;
    
    halfup=1;
    while(gradprof(halfup)), halfup=halfup+1; end
    halfdown=halfup;    % find midpoint of high gradient region
    while ~gradprof(halfdown),halfdown=halfdown+1; end
    if ~rimthick,rimthick=round(0.5*(halfdown-halfup)); end
    halfup=halfup+rimthick;
    outer(end+1,:)=[thiscol halfup];
    while gradprof(halfdown),halfdown=halfdown+1; end
%     nextup=halfdown;  % find midpoint of high gradient region
%     while ~gradprof(nextup),nextup=nextup+1; end
%     
    halfdown=halfdown+rimthick;
    %halfdown=0.5*(nextup+halfdown);
    inner(end+1,:)=[thiscol halfdown];
    halfup=numel(gradprof);
%     while gradprof(halfup),halfup=halfup-1; end %bottom of couch
%     while ~gradprof(halfup), halfup=halfup-1;end %inside couch
%     while gradprof(halfup),halfup=halfup-1; end %top of couch
%     while ~gradprof(halfup), halfup=halfup-1;end %above couch
    while gradprof(halfup),halfup=halfup-1; end  % outside ring edge
    halfdown=halfup;
    while ~gradprof(halfdown),halfdown=halfdown-1; end % ring body
    %halfup=0.5*(halfup+halfdown);  % midpoint of high gradient
    halfup=halfup-rimthick;
    outer(end+1,:)=[thiscol halfup];
    while gradprof(halfdown),halfdown=halfdown-1; end   % inside ring edge
%     nextup=halfdown;  % find midpoint of high gradient region
%     while ~gradprof(nextup),nextup=nextup-1; end
    %halfdown=0.5*(nextup+halfdown);
    halfdown=halfdown-rimthick;

    inner(end+1,:)=[thiscol halfdown];
    
    
end
thisrow=round(size(gradmask,1)/2);
gradprof=gradmask(thisrow,:);

halfup=1;
% find first surface of ring
while(gradprof(halfup)&&halfup<numel(gradprof)), halfup=halfup+1; end
halfdown=halfup;
% cross the surface
while ~gradprof(halfdown)&&halfdown<numel(gradprof),halfdown=halfdown+1; end
%halfup=0.5*(halfdown+halfup);
halfup=halfup+rimthick;
outer(end+1,:)=[halfup thisrow];
% find next surface
while gradprof(halfdown)&&halfdown<numel(gradprof),halfdown=halfdown+1; end
% nextup=halfdown;  % find midpoint of high gradient region
% while ~gradprof(nextup),nextup=nextup+1; end
%halfdown=0.5*(nextup+halfdown);
halfdown=halfdown+rimthick;
inner(end+1,:)=[halfdown thisrow];
halfup=numel(gradprof);;
while(gradprof(halfup)), halfup=halfup-1; end
halfdown=halfup;
while ~gradprof(halfdown),halfdown=halfdown-1; end
%halfup=0.5*(halfup+halfdown);
halfup=halfup-rimthick;
outer(end+1,:)=[halfup thisrow];
while gradprof(halfdown),halfdown=halfdown-1; end
% nextup=halfdown;  % find midpoint of high gradient region
% while ~gradprof(nextup),nextup=nextup-1; end
% halfdown=0.5*(nextup+halfdown);
halfdown=halfdown-rimthick;
inner(end+1,:)=[halfdown thisrow];
subplot(1,4,1);
plot(outer(:,1),outer(:,2),'+');
plot(inner(:,1),inner(:,2),'+');
subplot(1,4,2);
plot(outer(:,1),outer(:,2),'+');
plot(inner(:,1),inner(:,2),'+');
[xin,yin,rin]=circle_from_points(inner);
[xout,yout,rout]=circle_from_points(outer);
angles=[0:10:350]*pi()/180;
st=sin(angles);
ct=cos(angles);
outx=xout+rout*ct;
outy=yout+rout*st;
inx=xin+rin*ct;
iny=yin+rin*st;
subplot(1,4,1);
plot(inx,iny,'.');
plot(outx,outy,'.');
% find the ring from its geometry
xvals=1:size(midslice,2);
yvals=1:size(midslice,1);
[xmesh,ymesh]=meshgrid(xvals,yvals);
rsqmesh1=(xmesh-xin).*(xmesh-xin)+(ymesh-yin).*(ymesh-yin);
rsqmesh2=(xmesh-xout).*(xmesh-xout)+(ymesh-yout).*(ymesh-yout);
ringmask=(rsqmesh1>=rin*rin & rsqmesh2<=rout*rout);

subplot(1,4,2);
plot(inx,iny,'.');
plot(outx,outy,'.');

subplot(1,4,3);
hold off;
[airmask,legmask,pvsmask]=strip_peaks(midslice,rsqmesh1,rin,[],gca);

legvol(:,:,midplane)=legmask;
ringvol(:,:,midplane)=ringmask;
pvsvol(:,:,midplane)=pvsmask;
totmask=ringmask+2*legmask+3*pvsmask;
figure (plotfig); subplot(1,4,4);
imagesc(totmask),axis image, %colormap(gca,jet_black);
pause(0.05);
end