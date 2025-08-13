function [xcen,ycen,r] = circle_from_points(points)

npts=size(points,1);
xvec=[];
yvec=[];
for n=1:npts
    % form perpendicular bisector with next point
    nnext=n+1;
    if (nnext>npts), nnext=1; end
    x01=(points(nnext,1)+points(n,1))/2;  % midpoint
    y01=(points(nnext,2)+points(n,2))/2;
    
    dx1=points(nnext,1)-points(n,1);  % separation
    dy1=points(nnext,2)-points(n,2);
    r1=sqrt(dx1^2+dy1^2);   % length
    u1=dx1/r1;  % unit vectors
    v1=dy1/r1;
    nnext1=nnext+1;
    if(nnext1>npts);nnext1=1;end
    
    x02=(points(nnext1,1)+points(nnext,1))/2;  % midpoint
    y02=(points(nnext1,2)+points(nnext,2))/2;
    
    dx2=points(nnext1,1)-points(nnext,1);  % separation
    dy2=points(nnext1,2)-points(nnext,2);
    r2=sqrt(dx2^2+dy2^2);   % length
    u2=dx2/r2;  % unit vectors
    v2=dy2/r2;

    % solve for intersection with cramer's rule
    
    dmat=[-v1, v2; u1, -u2];
    cvec=[x02-x01; y02-y01];
    alphmat=dmat;
    alphmat(:,1)=cvec;
    betmat=dmat;
    betmat(:,2)=cvec;
    
    dd=det(dmat);
    alph=det(alphmat)/dd;
    bet=det(betmat)/dd;
    xvec=[xvec x01-alph*v1 x02-bet*v2];
    yvec=[yvec y01+alph*u1 y02+bet*u2];
end
xcen=mean(xvec);
ycen=mean(yvec);
rsum=0;
for n=1:npts
    dx=xcen-points(n,1);
    dy=ycen-points(n,2);
    rsum=rsum+sqrt(dx^2+dy^2);
end
r=rsum/npts;
