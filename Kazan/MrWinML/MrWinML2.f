!  MrWinML.f90 
!
!  FUNCTIONS:
!	MrWinML      - Entry point of console application.
!

!!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*
!
!  PROGRAM: MrWinML
!
!  PURPOSE:  Entry point for the console application.
!
!!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*

!	program MrWinML
	subroutine aaa

	implicit none

	! Variables


	! Body of MrWinML
* 
	INTEGER nkeys,ipos,ip,nwords,index,noccur,maclevel,ircode,mxval
	
	DOUBLE PRECISION AE, QE,gn
	
	COMMON /x1aR8/ AE(6,10),QE(6,10),gn(10) ! maximum number is 10
	INTEGER idebug,nnuc,mult
	LOGICAL x2
      
      COMMON /x1aI4/ idebug,nnuc,mult(10)
      COMMON /x1aL4/ x2(10)
	
	INTEGER XX,XY,XZ,YY,YZ,ZZ,X,Y,Z
	INTEGER nCnuc,cmult,Xtype
      PARAMETER(XX=1,XY=2,YY=3,XZ=4,YZ=5,ZZ=6)
	DOUBLE PRECISION  AEc, AT, GG, GT, Xfreq, Xwidth

	COMMON /corR8/ AEc(6,10),AT(6,10),GG(6),GT(6),Xfreq,Xwidth
      COMMON /corI4/ nCnuc,cmult(10),Xtype

	DOUBLE PRECISION B,tau,fmax
	INTEGER krid,nac
	PARAMETER (nac=512)
	INTEGER n,max,ifmt,npag
      REAL xmin,xmax
      DOUBLE PRECISION accu(nac)
****************************************	
*	simulation parameters
****************************************
*	ELECTRON SPIN SYSTEM
	!call eprreset
*	g-values 
	GT(XX) = 2.0 ! always diagonal ???
	GT(XY) = 2.0
	GT(XZ) = 2.0
	GT(YY) = 2.04
	GT(YZ) = 20.0
	GT(ZZ) = 2.1

	GG(XX) = 2.0 ! always diagonal ???
	GG(XY) = 0.0
	GG(XZ) = 0.0
	GG(YY) = 2.04
	GG(YZ) = 0.0
	GG(ZZ) = 2.1

      xtype = 0		! xtype ???
      xwidth = 200D0  ! unitary linewidth ??
      xfreq  = 1000D0 ! MW frequency, [MHz]
	
*	NUCLEAR SPIN SYSTEM	
*	number of nuclei
	nnuc = 1 ! maximum 10
*	HF tensor
	AE(1,1) = 0.666
	AE(2,1) = 0.666
	AE(3,1) = 0.666
	AE(4,1) = 0  ! alpha
	AE(5,1) = 0  ! beta
	AE(6,1) = 0  ! gamma
*	Quadrupole tensor
	QE(1,1) = 0.3
	QE(2,1) = 0.1
	QE(3,1) = -0.4
	QE(4,1) = 0  ! alpha
	QE(5,1) = 0  ! beta
	QE(6,1) = 0  ! gamma
*	nuclear g-value
	gn(1) = 0.404
*	nuclear pultiplicity
	mult(1) = 3 ! check .... 3=="S=1" ?
	cmult(1) = 3
*	EXPERIMENTAL PARAMETERS
	B = 3200 ! magnetic field [G]
	krid = 30 ! nKnots ??
	tau = 0.0 ! ==0 ->2pESEEM, otherwise 3pESEEM [ms]
	fmax = 10 ! nyquist frequency (1/[2dx])
	!nac = 100 ! nPoints ??
	
******* nac - data ********
	
	call eprlist
	call x1list(6)

	!call SETEPR(nwords,ip,maclevel,ircode)
	
	call x1a(B,krid,tau,fmax,accu,nac)
	
	!call simul(xmin,xmax,ifmt,accu,n,max,npag)
	!write(6, *) 'OK'
	call SETSIM(nwords,ip,maclevel,ircode)
	end 
	


********************************************************
********************************************************
************ ENTRY FOR MEX DLL *************************
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
********************************************************
c
	implicit none
	integer nExp, nSys, nOpt
      parameter (nExp = 10, nSys = 137, nOpt = 10)
	integer nlhs, nrhs
	integer plhs(nlhs), prhs(nrhs), mxGetM, mxGetN 
	integer mxIsNumeric, mxGetPr
      integer mxCreateDoubleMatrix
      integer x_pr, y_pr, Sys_pr, Exp_pr, Opt_pr
      integer m1, n1, m2, n2, size, mm, nn
      ! real*8 y, c
	!ALLOCATABLE ::  y(:, :)
c	real*8, pointer :: y
	real*8  Sy(nSys), Ex(nExp), Op(nOpt)
	integer nnpoints, nfpoints
	external mxGetPr
C  -----------------------------------------------------
C     Check for proper number of arguments. 
	
      if (nrhs .gt. 3) then
         call mexErrMsgTxt('Maximum of 3 input 
     .	   parameters required (Sys, Exp, Opt).')
	endif
      if (nlhs .ne. 1) then
         call mexErrMsgTxt('One output required (y).')
      endif

C     Check to see both inputs are numeric.
      if (mxIsNumeric(prhs(1)) .ne. 1) then
         call mexErrMsgTxt('Input # 1 is not a numeric.')
	endif
      if (mxIsNumeric(prhs(2)) .ne. 1) then
         call mexErrMsgTxt('Input #2 is not a numeric array.')
      endif
	if (mxIsNumeric(prhs(3)) .ne. 1) then
         call mexErrMsgTxt('Input #3 is not a numeric array.')
      endif
      
C     Check number of elemenys in input #1
      m1 = mxGetM(prhs(1))
      n1 = mxGetN(prhs(1))

      if(n1 .ne. 1 .or. m1 .gt. nSys) then
         call mexErrMsgTxt('Sys matrix is to big.')
      endif

C     Check number of elemenys in input #2
      m2 = mxGetM(prhs(2))
      n2 = mxGetN(prhs(2))
      if(n2 .ne. 1 .or. m2 .gt. nExp) then
         call mexErrMsgTxt('Exp matrix is not big enough.')
      endif     
C     Check number of elemenys in input #3
      m2 = mxGetM(prhs(3))
      n2 = mxGetN(prhs(3))
      if(n2 .ne. 1 .or. m2 .gt. nOpt) then
         call mexErrMsgTxt('Opt matrix is not big enough.')
      endif    	 
C  -------------------------------------------------------	

C     Create matrix for the return argument.
      !plhs(1) = mxCreateDoubleMatrix(NNpoints,1,0)


      ! y_pr = mxGetPr(plhs(1))
	
C     Load the data into Fortran arrays.
      Exp_pr = mxGetPr(prhs(2))
	call mxCopyPtrToReal8(Exp_pr,Ex,nExp)

      nnpoints = int(Ex(1)) ! number of points	
	nfpoints  = int(Ex(2))
	call linkroutine(nlhs,plhs,nrhs,prhs,nnpoints, nfpoints)

	
	end

************************************************************
	subroutine linkroutine(nlhs,plhs,nrhs,prhs,nnpoints, nFpoints)
	implicit none
	integer nrhs, nlhs, nnpoints, NFpoints
	double precision y(nnpoints, NFpoints)
	integer plhs(nlhs), prhs(nrhs), y_pr
	integer mxCreateDoubleMatrix, mxGetPr
	integer nExp, nSys, nOpt
	integer Sys_pr, Exp_pr, Opt_pr
      parameter (nExp = 10, nSys = 137, nOpt = 10)
	real*8  Sy(nSys), Ex(nExp), Op(nOpt)
	external mxGetPr


      Sys_pr = mxGetPr(prhs(1))
      Exp_pr = mxGetPr(prhs(2))
	Opt_pr = mxGetPr(prhs(3))

	call mxCopyPtrToReal8(Sys_pr,Sy,nSys)
	call mxCopyPtrToReal8(Exp_pr,Ex,nExp)
	call mxCopyPtrToReal8(Opt_pr,Op,nOpt)
		


	! allocate (y(nnpoints, 1))
      !y_pr = mxGetPr(plhs(1))
	  call f_MX(Sy, nSys, Ex, nExp, Op, nOpt, y, NNpoints,NFpoints)


*	  IF(Op(17).lt.1) GOTO 333
*	  call dispi('Len: ', size(y, 1), 5)
*	  call dispi('Len: ', size(y, 2), 5)
*	  call dispi('Plen: ', mm, 6)
* 333  continue	

C     Load the output into a MATLAB array.
      
	plhs(1)=mxCreateDoubleMatrix(nnpoints,nfpoints,0)
	y_pr=mxGetPr(plhs(1))

	call mxCopyReal8ToPtr(y,y_pr,NNpoints*nfpoints)

      
      end

****************************************************
	subroutine  f_MX(Sy, nSys, Ex, nExp, Op, nOpt, accu, 
     .	NNpoints, NFpoints)
	
	implicit none
	INTEGER PLHS(1)
	integer y_pr
	INTEGER nkeys,ipos,ip,nwords,index,noccur,maclevel,ircode,mxval
	integer nExp, nSys, nOpt, NNpoints, NFpoints, i, j
	DOUBLE PRECISION AE, QE,gn
	
	COMMON /x1aR8/ AE(6,10),QE(6,10),gn(10) ! maximum number is 10
	INTEGER idebug,nnuc,mult
	LOGICAL x2
      
      COMMON /x1aI4/ idebug,nnuc,mult(10)
      COMMON /x1aL4/ x2(10)
	
	INTEGER XX,XY,XZ,YY,YZ,ZZ,X,Y,Z
	INTEGER nCnuc,cmult,Xtype
      PARAMETER(XX=1,XY=2,YY=3,XZ=4,YZ=5,ZZ=6)
	DOUBLE PRECISION  AEc, AT, GG, GT, Xfreq, Xwidth
	real*8  Sy(nSys), Ex(nExp), Op(nOpt)

	COMMON /corR8/ AEc(6,10),AT(6,10),GG(6),GT(6),Xfreq,Xwidth
      COMMON /corI4/ nCnuc,cmult(10),Xtype

	DOUBLE PRECISION B, Bstart, Bend, Bstep,tau,fmax
	INTEGER krid,nac,nadd
	INTEGER n,max,ifmt,npag
      REAL xmin,xmax
      DOUBLE PRECISION accu(NNpoints, NFpoints), tempy(NNpoints, 1)

****************************************	
*	simulation parameters
****************************************
*	ELECTRON SPIN SYSTEM
	!call eprreset
*	g-values 
	GT(XX) = Sy(1) ! gxx
	GT(XY) = 0.0   ! gxy
	GT(XZ) = 0.0   ! gxz
	GT(YY) = Sy(2) !
	GT(YZ) = 0.0
	GT(ZZ) = Sy(3)

	GG(1) = Sy(1) ! gx
	GG(2) = Sy(2) ! gy
	GG(3) = Sy(3) ! gz
	GG(4) = Sy(4) ! alpha 
	GG(5) = Sy(5) ! beta
	GG(6) = Sy(6) ! gamma
	
	xwidth = Sy(7)  ! unitary linewidth ??
	do 9 i = 1, nnpoints
		tempy(i, 1) = 0.0
	do 8 j = 1, nfpoints
   8		accu(i, j) = 0.0
   9  continue

     
*	NUCLEAR SPIN SYSTEM	
*	number of nuclei
	nnuc = int(Sy(8)) ! maximum 10
	if(nnuc .gt. 10) then
         call mexErrMsgTxt('Number of nuclei is limited to 10')
      endif
*	HF tensor
	DO 10 i = 1 , nnuc

	nadd = (i-1)*14
	AE(1,i) = Sy(9+nadd)
	AE(2,i) = Sy(10+nadd)
	AE(3,i) = Sy(11+nadd)
	AE(4,i) = Sy(12+nadd)  ! alpha
	AE(5,i) = Sy(13+nadd)  ! beta
	AE(6,i) = Sy(14+nadd)  ! gamma
*	Quadrupole tensor
	QE(1,i) = Sy(15+nadd)
	QE(2,i) = Sy(16+nadd)
	QE(3,i) = Sy(17+nadd)
	QE(4,i) = Sy(18+nadd)  ! alpha
	QE(5,i) = Sy(19+nadd)  ! beta
	QE(6,i) = Sy(20+nadd)  ! gamma

	gn(i) = Sy(21+nadd)	   ! nuclear g-value
*	
	mult(i) = int(Sy(22+nadd)) ! nuclear mpultiplicity check .... 3=="S=1" ?
	cmult(i) = int(Sy(22+nadd))

  10	continue
*	EXPERIMENTAL PARAMETERS\
*	Ex(1) -> nnpoints
*     Ex(2) -> nFpoints
	Bstart = Ex(3) ! magnetic field [G] 
	Bend   = Ex(4)
	xfreq  = Ex(5) ! MW frequency, [MHz]
	tau = Ex(6) ! ==0 ->2pESEEM, otherwise 3pESEEM [ms]
	fmax = Ex(7) ! nyquist frequency (1/[2dx])
      xtype = Ex(8)		! xtype ???

	krid = int(Op(1))   ! nKnots ??
	idebug = int(Op(2)) ! show massages or not 
	
	!call eprlist
	!call x1list(6)

	!call SETEPR(nwords,ip,maclevel,ircode)
	Bstep = (Bend-Bstart)/Ex(2)
  	
	DO 20 i = 1 , nFpoints
		B = Bstart + Bstep*(i-1)
		call x1a(B,krid,tau,fmax,tempy,nnpoints)
		DO 21 j = 1, nnpoints
  			accu(j, i) = tempy(j, 1)
  21			tempy(j, 1) = 0.0 ! clean up the temporary storage
  20		continue
	!call mxCopyReal8ToPtr(accu, y_pr,NNpoints)
	
	END