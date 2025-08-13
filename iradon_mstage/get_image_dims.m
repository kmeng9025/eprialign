function [nendx,nendy,nendz,nendb]=get_image_dims(imtype,nbins);

if (imtype==1)
    nendb = nbins;
    nendx = nbins;
	nendy = nbins;
	nendz = nbins;
elseif (imtype==2)
    nendb = nbins;
	nendx = nbins;
	nendy = 1;
	nendz = nbins;
elseif (imtype==3)
    nendb = nbins;
	nendx = 1;
	nendy = nbins;
	nendz = nbins;
elseif (imtype==4)
    nendb = nbins;
	nendx = nbins;
	nendy = nbins;
	nendz = 1;
elseif (imtype==5)
    nendb = nbins;
	nendx = nbins;
	nendy = 1;
	nendz = 1;
elseif (imtype==6)
    nendb = nbins;
	nendx = 1;
	nendy = nbins;
	nendz = 1;
elseif (imtype==7) 
    nendb = nbins;
	nendx = 1;
	nendy = 1;
	nendz = nbins;
elseif (imtype==11) 
    nendb = 1;
    nendx = nbins;
	nendy = 1;
	nendz = nbins;
elseif (imtype==12) 
    nendb = 1;
    nendx = 1;
	nendy = nbins;
	nendz = nbins;
elseif (imtype==13) 
    nendb = 1;
    nendx = nbins;
	nendy = nbins;
	nendz = 1;
elseif (imtype==14) 
    nendb = 1;
    nendx = nbins;
	nendy = nbins;
	nendz = nbins;
else % imtype=={8,9,10} are not images
    nendb = nbins;nendx=1;nendy=1;nendz=1;
end    
