function [ dose ] = rgb_to_dose( film,r,g,b,d,varargin )
%RGB_TO_DOSE  converts RGB on radiochromic film to dose
%
% arguments:
% film:  the array of RGB values to be converted
% r,g,b:  the calibration arrays - R,G and B values at known doses
% d:    the dose points at which R,G,B are known.
% 
% varargin:  if first extra argument is 1, then the input RGB array will be
% converted to density by taking the log of R,G,B divided into 65535.  This
% assumes the R,G,B arrays are ALREADY in log units, which you will need to
% ensure by doing something like
%
% rd=log(65535./double(r));
% gd=log(65535./double(g));
% bd=log(65535./double(b));
% and passing rd, gd and bd into this function instead of r,g and b.
% if using XRQA2 film, you will want to ignore the blue channel which has
% little or no density in it and is very noisy.  you can force this by
% passing in a blue calibration array that is all zeros, in which case this
% function will set the blue channel of the input array to all zeros as
% well.
%
% C. Pelizzari April 2012
% 
dolog=0;
if (nargin >5), dolog= varargin{1}; end  %if 6th arg is 1, convert to density

sz=size(film);
% dose array will be same size as the input rgb array, but with only one
% element per point vs 3.
szd=sz;
szd(end)=1;
% will reshape the input array into a linear Nx3 vector to make this
% routine applicable to any sort of input - vectors, arrays of any
% dimension.
newdims=[prod(sz(1:end-1)),3]

film=reshape(film,newdims);
if(dolog), 
   film=log(65535./double(film));
end
if (b(1)==0) && (b(end)==0), film(:,3)=0; end

dose=zeros(newdims(1),1);
% tricolor_curve_dose2 can go faster if we precompute the 3D lengths and
% unit vectors of the segments of the RGB calibration curve
nsegs=size(r,1);
segs=zeros(nsegs,3);
for n=1:nsegs
    segs(n,:)=double([r(n) g(n) b(n)]);
end
[uvec1,len1]=poly_unit_vectors(segs);

for i = 1:newdims(1)
    if (mod(i,10000) == 0) disp(num2str(i)); end
        if (film(i,1)>max(r)), dose(i) = d(find(r==max(r))); 
        elseif (film(i,1) < min(r)), dose(i)=d(find(r==min(r)));
        else
           
        %uncomment the algorithm you want to use - tricolor_curve_dose2 is
        %the fastest, also the curve algorithms tend to have fewer big noise
        %spikes
        
        dose(i) = tricolor_curve_dose2(squeeze(film(i,:)),r,g,b,d,uvec1,len1);end
        %dose(i) = tricolor_curve_dose(squeeze(film(i,:)),r,g,b,d,uvec1,len1);end

        %dose(i) = tricolor_curve_dose(squeeze(film(i,:)),r,g,b,d);end
        %dose(i) = tricolor_dose(squeeze(film(i,:)),r,g,b,d);end
        %dose(i) = tricolor_micke_dose(squeeze(film(i,:)),r,g,b,d);end
end
% reshape the dose vector to match the size of the input array
dose=reshape(dose,szd);

end

