C     A MATLAB MEX routine for 4d serial backprojection
      
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
C-----------------------------------------------------------------------
C     (integer) Replace integer by integer*8 on the DEC Alpha and the
C     SGI 64-bit platforms
C
      integer plhs(*), prhs(*)
      integer mxCreateFull, mxGetPr
      integer sino_pr, nplr_pr, naz_pr, nspec_pr, nbins_pr, imtype_pr
	integer plri_pr, azii_pr, speci_pr, bc_pr
	integer	image_pr
	integer ndims
C-----------------------------------------------------------------------
      integer nlhs, nrhs
      integer mxGetM, mxGetN, mxIsNumeric
      integer m, n, size, image_size
	real*8 nplr, naz, nspec, nbins, imtype, plri, azii, speci, bc

C     Check for proper number of arguments. 
      if(nrhs .ne. 10) then
         call mexErrMsgTxt('10 inputs required. [sino,np,na,ns,
	1   plri,azii,speci,nb,bin_center,type]') 
	elseif(nlhs .ne. 1) then
         call mexErrMsgTxt('One output required. [image]')
      endif

C     Check to insure the input arrays are numeric (not strings).
      if(mxIsNumeric(prhs(1)) .eq. 0) then
         call mexErrMsgTxt('Input 1 must be a numeric array.')
      endif
      if(mxIsNumeric(prhs(2)) .eq. 0) then
         call mexErrMsgTxt('Input 2 must be a numeric array.')
      endif
      if(mxIsNumeric(prhs(3)) .eq. 0) then
         call mexErrMsgTxt('Input 3 must be a numeric array.')
      endif
      if(mxIsNumeric(prhs(4)) .eq. 0) then
         call mexErrMsgTxt('Input 4 must be a numeric array.')
      endif
      if(mxIsNumeric(prhs(5)) .eq. 0) then
         call mexErrMsgTxt('Input 5 must be a numeric array.')
      endif
      if(mxIsNumeric(prhs(6)) .eq. 0) then
         call mexErrMsgTxt('Input 6 must be a numeric array.')
      endif
      if(mxIsNumeric(prhs(7)) .eq. 0) then
         call mexErrMsgTxt('Input 7 must be a numeric array.')
      endif
	if(mxIsNumeric(prhs(8)) .eq. 0) then
         call mexErrMsgTxt('Input 8 must be a numeric array.')
      endif
	if(mxIsNumeric(prhs(9)) .eq. 0) then
         call mexErrMsgTxt('Input 9 must be a numeric array.')
      endif
	if(mxIsNumeric(prhs(10)) .eq. 0) then
         call mexErrMsgTxt('Input 10 must be a numeric array.')
      endif

C	Get the sinogram dimensions
	nplr_pr = mxGetPr(prhs(2))
	naz_pr = mxGetPr(prhs(3))
	nspec_pr = mxGetPr(prhs(4))
	nbins_pr = mxGetPr(prhs(8))
	imtype_pr = mxGetPr(prhs(10))

	call mxCopyPtrToReal8(nplr_pr,nplr,1)
	call mxCopyPtrToReal8(naz_pr,naz,1)
	call mxCopyPtrToReal8(nspec_pr,nspec,1)
	call mxCopyPtrToReal8(nbins_pr,nbins,1)
	call mxCopyPtrToReal8(imtype_pr,imtype,1)

C	Get the initial angles
	plri_pr = mxGetPr(prhs(5))
	azii_pr = mxGetPr(prhs(6))
	speci_pr = mxGetPr(prhs(7))
	bc_pr = mxGetPr(prhs(9))

	call mxCopyPtrToReal8(plri_pr,plri,1)
	call mxCopyPtrToReal8(azii_pr,azii,1)
	call mxCopyPtrToReal8(speci_pr,speci,1)
	call mxCopyPtrToReal8(bc_pr,bc,1)

C     Get the size of the sinogram.
      m = mxGetM(prhs(1))
      n = mxGetN(prhs(1))
      size = m*n
	
C     Sinogram must be of the given dimensions
      if(size.ne.(nplr*naz*nspec*nbins)) then
         call mexErrMsgTxt('Input dimensions are inconsistant.')
      endif
      
C     Create matrix for the return argument.
      sino_pr = mxGetPr(prhs(1))
	ndims=int((13.0-imtype)/3.0)
	if(imtype.eq.14) then
	   ndims=3
	endif
	image_size=1
	do 10 i=1,ndims
		image_size=nbins*image_size
10	continue 

C	image_size=nbins*nbins*nbins*nbins
      plhs(1) = mxCreateFull(image_size,1,0)
      image_pr = mxGetPr(plhs(1))
C      call mxCopyPtrToReal8(sino_pr,sino,size)

C     Call the computational subroutine.
      call bp_4ds(%val(sino_pr),nplr,naz,nspec,plri,azii,speci,nbins,
     1	bc,imtype,%val(image_pr))

C     Load the data into y_pr, which is the output to MATLAB
C      call mxCopyReal8ToPtr(image,image_pr,size)     

      return
      end

	subroutine bp_4ds(sino,nplr,naz,nspec,pri,ai,si,nbins,bc,imt,image)

	real*8 nplr, naz, nspec, pri, ai, si, nbins, bc, imt
	integer mtnang,mpnang,manang,kmd,imtype,nmd,ndimu,kdimu
	real*8 theta0,phi0,alpha0,center,fkhalf,fnhalf
	real, allocatable :: btemp1(:,:,:,:)
	real, allocatable :: btemp2(:,:,:,:)
	real*8 sino(*), image(*)
	character*20 string
	dimension ctt(nplr),stt(nplr)
	dimension ctf(naz),stf(naz)
	dimension cta(nspec),sta(nspec)


	mtnang=int(nplr)
	mpnang=int(naz)
	manang=int(nspec)
	kmd=int(nbins)
	imtype=int(imt)
	center=bc
	nmd=kmd
      ndimu=nmd
      kdimu=kmd

C	In our data acquisition the integral including the origin is nbins/2
C      fkhalf=float(kdimu)/2.
	fkhalf=center
      fnhalf=float(ndimu+1)/2.
C	Initialize array sizes
	nxsize=1
	nysize=1
	nzsize=1
	nbsize=ndimu
C	Start and end planes for B, which is spectral
      nstb=1
      nendb=ndimu
	IF (imtype.EQ.7) THEN 
			nstx = fnhalf
			nendx = fnhalf
			nsty = fnhalf
			nendy = fnhalf
			nstz = 1
			nendz = ndimu
			nzsize = ndimu
			nbfilt=0
			ntfilt=0
	ELSE IF (imtype.EQ.6) THEN
			nstx = fnhalf
			nendx = fnhalf
			nsty = 1
			nendy = ndimu
			nstz = fnhalf
			nendz = fnhalf
			nysize = ndimu
			nbfilt=0
			ntfilt=0
	ELSE IF (imtype.EQ.5) THEN
			nstx = 1
			nendx = ndimu
			nsty = fnhalf
			nendy = fnhalf
			nstz = fnhalf
			nendz = fnhalf
			nxsize = ndimu
			nbfilt=0
			ntfilt=0
	ELSE IF (imtype.EQ.4) THEN
			nstx = 1
			nendx = ndimu
			nsty = 1
			nendy = ndimu
			nstz = fnhalf
			nendz = fnhalf
			nxsize = ndimu
			nysize = ndimu
			nbfilt=1
			ntfilt=0
	ELSE IF (imtype.EQ.3) THEN
			nstx = fnhalf
			nendx = fnhalf
			nsty = 1
			nendy = ndimu
			nstz = 1
			nendz = ndimu
			nysize = ndimu
			nzsize = ndimu
			nbfilt=1
			ntfilt=0
	ELSE IF (imtype.EQ.2) THEN
			nstx = 1
			nendx = ndimu
			nsty = fnhalf
			nendy = fnhalf
			nstz = 1
			nendz = ndimu
			nxsize = ndimu
			nzsize = ndimu
			nbfilt=1
			ntfilt=0
	ELSE IF (imtype.EQ.1) THEN
			nstx = 1
			nendx = ndimu
			nsty = 1
			nendy = ndimu
			nstz = 1
			nendz = ndimu
			nxsize = ndimu
			nysize = ndimu
			nzsize = ndimu
			nbfilt=2
			ntfilt=1
	ELSE IF (imtype.EQ.14) THEN
			nstb = fnhalf
			nendb = fnhalf
			nstx = 1
			nendx = ndimu
			nsty = 1
			nendy = ndimu
			nstz = 1
			nendz = ndimu
			nxsize = ndimu
			nysize = ndimu
			nzsize = ndimu
			nbsize = 1
			nbfilt=0
			ntfilt=1
	END IF 
	
      pi=4.0*atan(1.0) 

C	Sets up the initial angles based on the image type
	dtheta=pi/float(mtnang)
	theta0=pi/2- pri
C      theta0=pi/2-.5*dtheta		
C     theta0=pi/2	
      dphi=pi/float(mpnang)
	phi0=pi/2- ai
C      phi0=pi/2-.5*dphi
C     phi0=pi/2
      dalpha=pi/float(manang)
	alpha0=pi/2- si
C      alpha0=pi/2-.5*dalpha

	IF (imtype.EQ.7) THEN 
			theta0 = 0.0
			phi0 = 0.0
			ndim = 2
	ELSE IF (imtype.EQ.6) THEN
			theta0 = pi/2
			phi0 = pi/2
			ndim = 2
	ELSE IF (imtype.EQ.5) THEN
			theta0 = pi/2
			phi0 = 0.0
			ndim = 2
	ELSE IF (imtype.EQ.4) THEN
			theta0 = pi/2
			phi0 = phi0
			ndim = 3
	ELSE IF (imtype.EQ.3) THEN
			theta0 = theta0
			phi0 = pi/2
			ndim = 3
	ELSE IF (imtype.EQ.2) THEN
			theta0 = theta0
			phi0 = 0.0
			ndim = 3
	ELSE IF (imtype.EQ.1) THEN
			theta0 = theta0
			phi0 = phi0
			ndim = 4
	ELSE IF (imtype.EQ.14) THEN
			alpha0 = pi/2
			theta0 = theta0
			phi0 = phi0
			ndim = 3
	END IF


C	Defines the theta angular interval and the initial angle. It saves all 
C		the sin and cos values. The initial angle is 90-.5dtheta.
C		theta is incremented by -dtheta
      do 17 i=1,mtnang
      theta=theta0-float(i-1)*dtheta
	theta=pi/2-theta  
      ctt(i)=cos(theta)	 
17    stt(i)=sin(theta) 

C	Defines the phi anglular interval and the initial angle. It saves all 
C		the sin and cos values. The initial angle is 90-.5dphi.
C		theta is incremented by -dphi
      do 18 i=1,mpnang
      phi=phi0-float(i-1)*dphi
C	phi=pi/2+phi
      ctf(i)=cos(phi)
 18   stf(i)=sin(phi)

C	Defines the alpha anglular interval and the initial angle. It saves all 
C		the sin and cos values. The initial angle is 90-.5dalpha.
C		alpha is incremented by -dalpha
      do 19 i=1,manang
      alpha=alpha0-float(i-1)*dalpha
	alpha=pi/2-alpha
      cta(i)=cos(alpha)
 19   sta(i)=sin(alpha)

	allocate(btemp1(mtnang,mpnang,nbsize,ndimu))

	call mexPrintf('Entering first loop')
	call mexCallMATLAB(0,NULL,0,NULL,"new_line")

C	write(string,'("manang ",I)') manang
C	call mexPrintf(string)
C	write(string,'("mtnang ",I)') mtnang
C	call mexPrintf(string)
C	write(string,'("mpnang ",I)') mpnang
C	call mexPrintf(string)
C	call mexCallMATLAB(0,NULL,0,NULL,"new_line")
C	write(string,'("nbsize ",I)') nbsize
C	call mexPrintf(string)
C	write(string,'("ndimu ",I)') ndimu
C	call mexPrintf(string)
C	call mexCallMATLAB(0,NULL,0,NULL,"new_line")


C	Below, the origin for i* indices is a cube corner
C		the origin for * indices is the cubic center
	do 100 it=1,mtnang
	do 110 ip=1,mpnang
      do 120 ib=nstb,nendb
      b=-fnhalf+float(ib)
	do 130 is=1,ndimu
      s=-fnhalf+float(is)
C	write(string,'("s",F)') s
C	call mexPrintf(string)
C	write(string,'("ib",I)') ib
C	call mexPrintf(string)      
C	write(string,'("is",I)') is
C	call mexPrintf(string)
C	write(string,'("it",I)') it
C	call mexPrintf(string)
C	write(string,'("ip",I)') ip
C	call mexPrintf(string)
	iib=ib-nstb+1
      btemp1(it,ip,iib,is)=0.0 
C	write(string,'("*")') 
C	call mexPrintf(string)
      

      do 140 ia=1,manang
	sca=s*cta(ia)
	bsa=b*sta(ia)
	xi=sca+bsa
      if(abs(xi).lt.fkhalf) then  
        xi=xi+fkhalf   
        kxi=xi
C     write(string,'("kxi",I)') kxi
C	call mexPrintf(string)
  	  if(kxi.lt.1.or.kxi.gt.kdimu-1) go to 140  
        dist=xi-kxi 
	  index=kxi+(ia-1)*kmd+(ip-1)*manang*kmd+(it-1)*manang*mpnang*kmd
C	write(string,'("index",I)') index
C	call mexPrintf(string)
C	call mexCallMATLAB(0,NULL,0,NULL,"new_line")

        term=sino(index)*(1.0-dist)+sino(index+1)*dist 
	  term=term/manang
	  do 150 i=1,nbfilt
		term=term*abs(cta(ia))
 150    continue 
        btemp1(it,ip,iib,is)=btemp1(it,ip,iib,is)+term 
C        btemp1(it,ip,ib,is)=btemp1(it,ip,ib,is)+term*cta(ia)*cta(ia) 
      endif 
 140  continue 
 130  continue 
 120  continue
 110  continue 
 100  continue

	allocate(btemp2(mpnang,ndimu,nzsize,nbsize))

	call mexPrintf(', second loop')

C 	Below, the origin for i* indices is a cube corner
C		the origin for * indices is the cubic center
	do 200 ip=1,mpnang
	do 210 ib=1,nbsize
      do 220 iz=nstz,nendz
      z=-fnhalf+float(iz)
	do 230 is=1,ndimu
      s=-fnhalf+float(is)
      
	iiz=iz-nstz+1

      btemp2(ip,is,iiz,ib)=0.0 

      do 240 it=1,mtnang
	 sct=s*ctt(it)
	 zst=z*stt(it)
	 xi=sct+zst
      if(abs(xi).lt.fnhalf) then  
        xi=xi+fnhalf   
        nxi=xi
        if(nxi.lt.1.or.nxi.gt.ndimu-1) go to 240  
        dist=xi-nxi 
        term=btemp1(it,ip,ib,nxi)*(1.0-dist)+btemp1(it,ip,ib,nxi+1)*dist 
	  term=term/mtnang
	  do 250 i=1,ntfilt
		term=term*abs(ctt(it))
 250    continue 
	  btemp2(ip,is,iiz,ib)=btemp2(ip,is,iiz,ib)+term
C	  btemp2(ip,is,iiz,ib)=btemp2(ip,is,iiz,ib)+term*abs(ctt(it))
	endif
 240  continue 
 230  continue 
 220  continue
 210  continue 
 200  continue

	deallocate(btemp1);

	call mexPrintf(', third loop')

C	Below, the origin for i* indices is a cube corner
C		the origin for * indices is the cubic center
	do 300 ib=1,nbsize
	do 310 iz=1,nzsize
      do 320 ix=nstx,nendx
      x=-fnhalf+float(ix)
	do 330 iy=nsty,nendy
      y=-fnhalf+float(iy)
      
	iix=ix-nstx+1
	iiy=iy-nsty+1

	index=iix+(iiy-1)*nxsize+(iz-1)*nxsize*nysize+
     1	(ib-1)*nxsize*nysize*nzsize

      image(index)=0.0 

      do 340 ip=1,mpnang
       xcf=x*ctf(ip)
       ysf=y*stf(ip)
       xi=ysf+xcf 
      if(abs(xi).lt.fnhalf) then  
        xi=xi+fnhalf   
        nxi=xi
        if(nxi.lt.1.or.nxi.gt.ndimu-1) go to 340  
        dist=xi-nxi 
        term=btemp2(ip,nxi,iz,ib)*(1.0-dist)+btemp2(ip,nxi+1,iz,ib)*dist 
        term=term/mpnang
	  image(index)=image(index)+term
      endif 
 340  continue 
 330  continue 
 320  continue
 310  continue 
 300  continue

	call mexPrintf(', DONE.')

	deallocate(btemp2);
	return
      end
