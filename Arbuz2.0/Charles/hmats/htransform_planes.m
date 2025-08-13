function newpoints = htransform_planes(pointlist, hmatlist)

nplanes = size(pointlist, 1);
npoints = size(pointlist, 3);

newpoints=zeros(size(pointlist));

for i = 1: nplanes
    mypointlist = squeeze(pointlist(i,:,:));
    rotpts = htransform_vectors(hmatlist(:,:,i), mypointlist');
    newpoints(i,:,:) = rotpts';
end