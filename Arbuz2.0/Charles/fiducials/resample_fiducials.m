function samples = resample_fiducials(lines, planenorm, locations)

npts=size(lines,1);
nfids = npts/2;
nlocs=prod(size(locations));
samples = zeros(nlocs, 3, nfids);

projections = lines * reshape(planenorm, prod(size(planenorm)), 1);

thisfid=0;
for myfid=1:2:npts
    thisfid=thisfid+1;
    p1=projections(myfid);
    p2=projections(myfid+1);
    
    l1=lines(myfid,:);
    l2=lines(myfid+1,:);
    
    needed=[locations - repmat(p1,size(locations,1),1)]/(p2-p1);
    
    for here=1:nlocs;
        samples(here,:,thisfid) = l1+needed(here)*(l2-l1);
    end
end