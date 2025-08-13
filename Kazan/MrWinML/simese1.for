*---------------------------------------------------------------------
      SUBROUTINE SETSIM(nwords,ip,maclevel,ircode)
*---------------------------------------------------------------------
*
*     SET INFO ON NUCLEI AND EXPERIMENTAL PARAMETERS
*
*     NUCLEUS    > MULT ( mult, x1/x2 )
*                > GN   ( gn )  {1H:5.58536}{2D:0.857387}
*                               {13C:1.4044}{14N:0.40347}
*                               {15N:-.56596}
*                > A    ( a1,a2,a3,al,be,ga)
*                > Q    ( a1,a2,a3,al,be,ga)
*                > LIST
*                > ADDNUC    ( inuc )   0: new nuc; i=index
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER nkeys,ipos,ip,nwords,index,noccur,maclevel,ircode,mxval
      PARAMETER(nkeys=11,mxval=20)
      LOGICAL ok
      CHARACTER*10 menu(nkeys)
      CHARACTER*60 help(nkeys)
*---> global
      INTEGER merror
      COMMON /macro/  merror
*---> local
      DOUBLE PRECISION GN,A(6),Q(6),rval(mxval)
      INTEGER mult,ival(mxval),vindexR(mxval),vindexI(mxval),inuc,
     .        narg
      LOGICAL x2
      DATA gn,a,q /13*0.D0/, mult /3/
      DATA MENU /'MULT      ','GN        ','A-TENSOR  ','Q-TENSOR  ',
     .           'LIST      ','ADD       ',
     .           'RESET     ','CONFIG    ','EXP       ','HELP      ',
     .           'RETURN    '/
      DATA HELP
     ./' syntax: multiplicity number_of_equivalent_nuclei [1,2]     ',
     . ' syntax: Nuclear_G-value                                    ',
     . ' syntax: A(1) A(2) A(3) Alpha Beta Gamma                    ',
     . ' syntax: Q(1) Q(2) Q(3) Alpha Beta Gamma                    ',
     . ' syntax:                                                    ',
     . ' syntax: [index of nucleus]                                 ',
     . '         (clear tensor info)                                ',
     . '         (configure integration process)                    ',
     . ' syntax: Krid Npoints Field Tau Fmax                        ',
     . ' Produces this text                                         ',
     . ' (Exit)                                                     '/
      ipos = ip
      ircode = 0
    1 continue
      ok = .true.
      goto(2,3,3) maclevel
    2 continue
*----> interpretation/definition level:
        IF(ip.eq.0) THEN
           call message(ircode)
           !write(6,*) ' GETNUC>'
           CALL macin(nwords,maclevel)
           ipos = 1
        ENDIF
        call recog(ipos,menu,nkeys,index,noccur)
        if(noccur.ne.1) goto 1313
        goto 9
*-------------> execute level <----------
    3 continue
      call mgetcode(index)
      if(merror.ne.0) then
           !write(6,*) ' MACRO EXECUTION ABORTED, error-code=', merror
           maclevel = 0
           goto 1313
      endif
    9 continue
*=======>>> COMMAND DISPATCHER <<<<<===========
      goto(10,20,30,40,50,60,70,80,90,100,110)index
      !write(6,*) ' illegal code SETSIM',index
      stop
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*--->  MULTIPLICITY [ MULT, X1/X2: TWO EQUIVALENT NUCLEI ]
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   10 continue
      call macparse(nwords,ipos+1,maclevel,index,2,0,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,2)
      call chkint(ival(1),2,8,ok)
      call chkint(ival(2),1,2,ok)
      if(.not.ok) goto 1313
      x2 = ival(2).eq.2
      mult = ival(1)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> GN enter nuclear g-factor
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   20 continue
      call macparse(nwords,ipos+1,maclevel,index,0,1,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefR(rval,vindexR,1)
      gn = rval(1)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> A-TENSOR (A1,A2,A3,alpha,beta,gamma)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   30 continue
      call macparse(nwords,ipos+1,maclevel,index,0,6,ival,A,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefR(A,vindexR,6)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> Q-TENSOR (A1,A2,A3,alpha,beta,gamma)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   40 continue
      call macparse(nwords,ipos+1,maclevel,index,0,6,ival,Q,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefR(Q,vindexR,6)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> LIST information of nuclei
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   50 continue
      call x1list(6)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> ADD nucleus to x1-module
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   60 continue
      if(maclevel.le.1) then
         inuc = 0
         if(nwords.gt.ipos) call getint(ipos+1,inuc,ok)
         if(.not.ok) goto 1313
      endif
      if(maclevel.eq.1) then
         call mputcode(index)
         ival(1) = inuc
         vindexI(1) = 0
         call mputint(ival,vindexI,1)
         goto 4711
      else if(maclevel.ge.2) then
         call mgetint(ival,vindexI,1)
         inuc = ival(1)
      endif
      call x1addnuc(mult,gn,A,Q,x2,inuc,ok)
      if(.not.ok) goto 1313
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> RESET (nuclear hamiltonian parameters)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   70 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call x1reset(0)
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> CONFIG set configuration of simulation module
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   80 continue
      !write(6,*) ' config cummod'
      IF(nwords-ipos.lt.6) THEN
         call getcum(rval(1),rval(2),ival(1),ival(2),ival(3))
         !write(6,188) rval(1),rval(2),ival(1),ival(2),ival(3)
  188    format(' ACCUM: threshini[',F6.4,'] threshlim[',F6.4,']'/
     .       '        limtable[',I3,'] method[',I1,'] debug[',I1,']')
      ELSE
         call getrray(ipos+1,rval,vindexR,2,ok)
         call getiray(ipos+3,ival,vindexI,3,ok)
         call mderefR(rval,vindexR,2)
         call mderefI(ival,vindexI,2)
         ok = .true.
         call chkreal(rval(1),0.D0,1.D0,ok)
         call chkreal(rval(2),0.D0,1.D0,ok)
         call chkint (ival(1),1,1000 ,ok)
         call chkint (ival(2),1,2    ,ok)
         call chkint (ival(3),0,1    ,ok)
         if(.not.ok) goto 1313
         call setcum(rval(1),rval(2),ival(1),ival(2),ival(3))
      ENDIF
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> EXP    set experimental parameters
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   90 continue
*     !write(6,*) ' SET EXP. PARAMETERS',maclevel
      call macparse(nwords,ipos+1,maclevel,index,2,3,ival,rval,
     .              vindexI,vindexR,narg)
      IF(narg.lt.0) goto 1313
      IF(narg.gt.0) THEN
         call mderefI(ival,vindexI,2)
         call mderefR(rval,vindexR,3)
         call setexp(rval(1),rval(2),rval(3),ival(1),ival(2))
      ELSE
         call listexp(6)
         goto 1515
      ENDIF
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> help list
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  100 continue
      call helplst(menu,help,nkeys)
*     call comlist(menu,nkeys)
      !write(6,*)
      !' mult/gn/x2     A         euler         Q         euler'
      !write(6,'(I4,7X,4(1X,G11.4))') mult, A(1), A(4), Q(1), Q(4)
      !write(6,'(G11.4,4(1X,G11.4))') gn  , A(2), A(5), Q(2), Q(5)
      !write(6,'(L4,7X,4(1X,G11.4))') x2  , A(3), A(6), Q(3), Q(6)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  110 continue
*---> TERMINATION
 4711 ircode = 0
      goto 9999
 1515 ircode = 0
      if(maclevel.eq.1) ircode = 1
      goto 9999
 1313 continue
      ircode = -1
 9999 continue
      if(ircode.ne.0.and.maclevel.eq.1) call rmcode
      if(ip.eq.0) goto 1
      END

*----------------------------------------------------------------------
*     INTERFACING SECTION
*----------------------------------------------------------------------
      SUBROUTINE simul(xmin,xmax,ifmt,accu,n,max,npag)
*----------------------------------------------------------------------
*     INTERFACE between data-page-driver and
*     ======>>> x1a-powder ESEEM simulation routine <<<<<=======
*     accu: real*8 accumulation buffer
*     max:  n1-dimension of page-system
*     npag: number of pages available
*     output:
*     n:    number of points in buffer
*     xmin/xmax x-axis limits
*     ifmt: format(-8:double/8:complex/4:real)
*---------------------------------------
      IMPLICIT none
      INTEGER n(*),max,ifmt(*),npag,junit
      REAL xmin(*),xmax(*)
      DOUBLE PRECISION accu(max,*)
*-->local
      INTEGER xkrid,krid,xnac,nac
      DOUBLE PRECISION xB,B,xtau,tau,xfmax,fmax
      DATA B/3300.0/,tau/0.2/,fmax/10.0/,krid/5/,nac/512/
*
      if(fmax.le.0.D0.or.nac.le.0.or.nac.gt.max)
     .  stop ' illegal argument in x1a'
      CALL x1a(B,krid,tau,fmax,accu,nac)
***** CALL xxxaa(B,krid,tau,fmax,accu,nac,max,npag)
      CALL r8r4(accu,accu,nac)
      xmin(1) = 0.0
      xmax(1) = fmax
      ifmt(1) = 4
      n(1)    = nac
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY setexp(xB,xtau,xfmax,xkrid,xnac)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      B = xB
      krid = xkrid
      tau  = xtau
      fmax = xfmax
      nac  = xnac
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY listexp(junit)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      !write(junit,
      ! '('' FIELD='',G10.3,'' TAU='',G10.3,'' FMAX='',G10.3)')B,tau,fmax
      !write(junit,'('' KRID ='',I10  ,'' NAC='',I10)') krid,nac
      END

***********************************************************************
*
* ESEEMX1 Simulation module:
*
* simulation of S=1/2 and arbitrary number of nuclear spins
*
* current implementation I=1/2, I=1, max number of nuc: maxnuc=10
*                        2-pulse and 3-pulse ESEEM (1D)
*
* Contains: interfacing routines to arbitrary main program
*           (initialize, communication, key-simulation routine)
*           (dedicated numerical procedures)
* Uses:     MRLIB1 (general spin Hamiltonian routines, constants)
*           LUXLIB1: utilities (sorting, io ...)
*
***********************************************************************
      SUBROUTINE x1reset(idebugx)
*----------------------------------------------------------------------
* initialize X1 experimental module COMMON blocks
*
      IMPLICIT none
****v
      INTEGER maxnuc,nnuc,mult,maxmul,idebug,junit,i,j
      PARAMETER (maxnuc=10,maxmul=8)
      DOUBLE PRECISION AE,QE,gn
      LOGICAL x2
      COMMON /x1aR8/ AE(6,maxnuc),QE(6,maxnuc),gn(maxnuc)
      COMMON /x1aI4/ idebug,nnuc,mult(maxnuc)
      COMMON /x1aL4/ x2(maxnuc)
****^
      INTEGER idebugx
      idebug = idebugx
      CALL dzero(AE,6*maxnuc)
      CALL dzero(QE,6*maxnuc)
      CALL dzero(gn,maxnuc)
      CALL izero(mult,maxnuc)
      CALL lzero(x2,maxnuc)
      nnuc = 0
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY x1list(junit)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if(nnuc.eq.0) return
      DO 20 i = 1 , nnuc
      !write(junit,
      !'('' Nucl. nr.'',I2,''  mult='',I2,''  x2='',L2,'' gn='',G11.4)')
      !                i, mult(i), x2(i), gn(i)
      !write(junit,'('' A='',6(1X,G11.4))') (AE(j,i), j = 1 , 6)
      !write(junit,'('' Q='',6(1X,G11.4))') (QE(j,i), j = 1 , 6)
  20  continue
      END
*
*----------------------------------------------------------------------
      SUBROUTINE x1addnuc(mulx,gnx,AEx,QEx,x2x,inuc,ok)
*----------------------------------------------------------------------
* Add info on nuclear spin
*
      IMPLICIT none
****v
      INTEGER maxnuc,nnuc,mult,maxmul,idebug
      PARAMETER (maxnuc=10,maxmul=8)
      DOUBLE PRECISION AE,QE,gn
      LOGICAL x2
      COMMON /x1aR8/ AE(6,maxnuc),QE(6,maxnuc),gn(maxnuc)
      COMMON /x1aI4/ idebug,nnuc,mult(maxnuc)
      COMMON /x1aL4/ x2(maxnuc)
****^
      INTEGER mulx,inuc
      DOUBLE PRECISION AEx(6),QEx(6),gnx
      LOGICAL x2x,ok
      ok = .false.
      if(inuc.gt.nnuc.or.inuc.lt.0) return
      if(inuc.eq.0) then
         nnuc = nnuc + 1
         inuc = nnuc
      endif
      if(inuc.gt.maxnuc) stop ' too many nuclei: > 10 '
      if(mulx.gt.maxmul) stop ' multiplicity too large: > 8 '
      mult(inuc) = mulx
      gn(inuc) = gnx
      x2(inuc) = x2x
      call vcopy(AEx,AE(1,inuc),1.D0,6)
      call vcopy(QEx,QE(1,inuc),1.D0,6)
      ok = .true.
      END

*----------------------------------------------------------------------
      SUBROUTINE x1a(B,krid,tau,fmax,accu,nac)
*----------------------------------------------------------------------
*      KEY ROUTINE
*          --->>> X1A Experiment driver <<<---
*     Simulate ESEEM experiment with parameters:
*     B = Magnetic field (GAUSS)
*     nu= microwave frequency (if = 0) then isotropic G assumed
*         and excitation on resonance
*     krid: powder integration grid
*     tau: in three pulse ESEEM experiment (0: two-pulse experiment)
*     fmax: nyquist frequency of time domain
*     accu: accumulator for stick spectrum
*-----
      IMPLICIT none
***v
      INTEGER maxnuc,nnuc,mult,maxmul,idebug
      PARAMETER (maxnuc=10,maxmul=8)
      DOUBLE PRECISION AE,QE,gn
      LOGICAL x2
      COMMON /x1aR8/ AE(6,maxnuc),QE(6,maxnuc),gn(maxnuc)
      COMMON /x1aI4/ idebug,nnuc,mult(maxnuc)
      COMMON /x1aL4/ x2(maxnuc)
***^
*
*---> arguments
      INTEGER krid,nac
      DOUBLE PRECISION B,tau,fmax,accu(0:*)
*
*--   >>>local
      INTEGER maxc,maxtr,ncalls,nselect
      PARAMETER (maxc=500,maxtr=64)
      DOUBLE PRECISION factor,cumint
*--   >>>hamiltonian parameters
      DOUBLE PRECISION A(6,maxnuc),Q(3,maxnuc),R(3,3,maxnuc)
*--   >>>Spin matrices (general spin system)
      DOUBLE PRECISION Sz(maxmul,maxnuc),Sx(maxmul,maxnuc),
     .                 SSxy(maxmul,maxnuc),SzS(maxmul,maxnuc)
*--   >>>magnetic field parameters
      DOUBLE PRECISION k(3),L(3),theta,phi,th,ph,pi
*
*--   >>>frequencies and intensities
      DOUBLE PRECISION ac(maxc),fa(maxtr),fb(maxtr),aa(maxtr),ab(maxtr),
     .                          a0,a0a,a0b,atab(0:maxc),ftab(0:maxc),
     .    ataba(0:maxtr),ftaba(0:maxtr),atabb(0:maxtr),ftabb(0:maxtr)
      EQUIVALENCE (ataba(1),aa),(ftaba(1),fa)
      EQUIVALENCE (atabb(1),ab),(ftabb(1),fb)
      INTEGER i,nc,index(2,maxc),ntab,ntran
*--   >>>experiment
      LOGICAL xlast
      DATA ftaba /0.D0,maxtr*0.D0/,ftabb/0.D0,maxtr*0.D0/
      DATA ataba /0.D0,maxtr*0.D0/,atabb/0.D0,maxtr*0.D0/
      DATA pi /3.141592653589793D0/
*
*--------------------------------
*
      if(nnuc.le.0) return
*
      DO 10 i = 1 , nnuc
      call euler2cos(AE(4,i),R(1,1,i))
      call Trot(AE(1,i),R(1,1,i),A(1,i))
      call euler2cos(QE(4,i),R(1,1,i))
      call vcopy(QE(1,i),Q(1,i),1.d0,3)
  10  CONTINUE
      call spinZF(Sx,Sz,SSxy,SzS,Q,mult,maxmul,nnuc)
*
*--   initialize accumulation/combination module
*****>>>>  inicum(nc,fmax,thold,tholdx,ltab,idebug)
*     CALL inicum(nac,fmax,0.005D0,0.9D0,50,idebug)
      CALL inicum(nac,fmax)
*
*---  POWDER INTEGRATION
      ncalls = 0
      nselect= 0
      cumint = 0D0
  30  CONTINUE
      call pwdpeal(krid,theta,phi,ncalls,xlast)
      !if(idebug.eq.1) write(6,'(2(1x,G12.5))') theta,phi
      !if(idebug.eq.1) write(6,'(2(1x,G12.5))') accu(0),accu(1)
      th = theta*pi/180.D0
      ph = phi  *pi/180.D0
      k(1)=DSIN(th)*DCOS(ph)
      k(2)=DSIN(th)*DSIN(ph)
      k(3)=DCOS(th)
*
      call eprselect(B,K,L,factor)
      if(factor.lt.1D-4) goto 21
      DO 20 i = 1 , nnuc
        ntran = mult(i)*(mult(i)-1)/2
        CALL ESEMIMS(GN(i),mult(i),A(1,i),Q(1,i),R(1,1,i),
     .               Sz(1,i),Sx(1,i),SSxy(1,i),SzS(1,i),
     .               B,K,L,FA,FB,a0,aa,ab,ac,index,maxc,nc)
*       CALL ESEMIMS(GN(i),mult(i),A(1,i),Q(1,i),R(1,1,i),
*    .               B,K,L,FA,FB,a0,aa,ab,ac,index,maxc,nc)
        IF (tau.lt.0.d0) THEN
          CALL ese2p(fa,fb,a0,aa,ab,ac,index,ntran,nc,atab,ftab,ntab)
          !if(idebug.eq.1) write(6,*) ' ntran, nc, ntab',ntran,nc,ntab
          CALL nuccum(atab,ftab,ntab,1,x2(i),i,nnuc,accu,nac,factor)
        ELSE
          a0a = a0
          a0b = a0
          CALL ese3psup(fa,fb,a0a,a0b,aa,ab,ac,index,ntran,nc,tau)
          ataba(0) = a0a
          atabb(0) = a0b
          ntab = ntran+1
          CALL nuccum(ataba,ftaba,ntab,1,x2(i),i,nnuc,accu,nac,factor)
          CALL nuccum(atabb,ftabb,ntab,2,x2(i),i,nnuc,accu,nac,factor)
        ENDIF
  20  CONTINUE
      nselect = nselect+1
      cumint  = cumint + factor
  21  CONTINUE
      if(.not.xlast) goto 30
*---
      if(ncalls.le.0) return
*     write(6,*) ' orientations:', ncalls
*     write(6,*) ' selections  :', nselect
*     write(6,*) ' relative int:', cumint/DBLE(ncalls)
      do 40 i = 0 , nac-1
  40   accu(i) = accu(i) *0.5D0 / dble(ncalls)
  	! write(6,'(G11.4)') accu(i)
      END
*----------------------------------------------------------------------
      SUBROUTINE ESEMIMS(GN,mult,A,Q,R, Sz,Sx,SSxy,SzS,
     -                           B,K,L,fa,fb,a0,aa,ab,ac,index,maxc,nc)
*----------------------------------------------------------------------
*     calculate basic modulation intensities for ESEEM experiment
*
* >>  input:
*     GN       : nuclear g-factor
*     MULT     : nuclear multiplicity
*     A(6)     : hyperfine matrix (lower triangle)
*     Q(3)     : quadrupole tensor (principle values)
*     R(3,3)   : eigenvector matrix of Q-tensor
*     B        : magnitude of Magnetic field
*     K(3)     : orientation vector of B-field
*     L(3)     : orientation vector of local field (anisotr. g)
* >>  output:
*     FA,FB    : alpha, beta frequencies
*     AA,AB,AC : alpha, beta and combination intensities
*     A0       : DC-component
*    index(2,i): index of combination frequencies (1,i) -> a-freq
*                                                 (2,i) -> b-freq
*     NC       : number of combination frequencies
*     MAXC     : max number of combination frequencies
*
*  units: time (us), frequency(MHz), energie(10-4cm-1), field(gauss)
*
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      IMPLICIT none
*  >> arguments
      INTEGER nc,maxc,mxm,mxt,mult,index(2,maxc)
      PARAMETER (mxm=8,mxt=64)
      DOUBLE PRECISION GN,A(6),Q(3),R(3,3),B,K(3),L(3),
     .                 FA(*),FB(*),A0,AA(*),AB(*),AC(*)
*  >> input hypfield
      DOUBLE PRECISION BX,BY,BZ,LX,LY,LZ,HX(2),HY(2),HZ(2)
*  >> input spin1
      DOUBLE PRECISION EA(mxm),EB(mxm),VR(mxt,2),VI(mxt,2)
*  >> input spinZF and spinZE
      DOUBLE PRECISION Sz(mxm),Sx(mxm),SSxy(mxm),SzS(mxm)
*  >> input inprod,complx
      DOUBLE PRECISION Rprob(mxt),Iprob(mxt),Mprob(mxt),Fprob(mxt)

      IF(mult.eq.2) THEN
         CALL protese(B,k,l,gn,A,fa(1),fb(1),a0,aa(1),ab(1),ac(1))
         nc = 1
         index(1,1) = 1
         index(2,1) = 1
         RETURN
      ENDIF
*     IF(mult.ne.3)
*    .   stop ' ESEMIMS other mult than 2 or 3 not yet implemented'
      BX=B*K(1)
      BY=B*K(2)
      BZ=B*K(3)
      LX=L(1)
      LY=L(2)
      LZ=L(3)
      CALL hypfield(bx,by,bz,lx,ly,lz,gn,A,hx,hy,hz)
*
      IF(mult.eq.3) THEN
        CALL spin1(hx(1),hy(1),hz(1),Q,R,EA,Vr(1,1),Vi(1,1))
        CALL spin1(hx(2),hy(2),hz(2),Q,R,EB,Vr(1,2),Vi(1,2))
*     ENDIF
      ELSE
        call spinZE(Sx,Sz,SSxy,SzS,mult,hx(1),hy(1),hz(1),R,
     .              Rprob,Iprob,Mprob,EA,VR(1,1),VI(1,1))
        call spinZE(Sx,Sz,SSxy,SzS,mult,hx(2),hy(2),hz(2),R,
     .              Rprob,Iprob,Mprob,EB,VR(1,2),VI(1,2))
      ENDIF
*
      CALL inprod(Vr(1,1),Vi(1,1),Vr(1,2),Vi(1,2),Rprob,Iprob,mult)
      CALL complx(Rprob,Iprob,Mprob,Fprob,mult*mult,1)
      CALL mims23(Mprob,Fprob,mult,a0,aa,ab,ac,index,nc,maxc)
      CALL genfreq(EA,EB,fa,fb,mult)
      END

*------------------------------------------------------------------
      SUBROUTINE MIMS23(tr,trphi,mult,chi0,aa,ab,ac,index,nc,maxc)
*------------------------------------------------------------------
*     Calculation of 2-and 3-pulse ESEEM intensities according to
*     the expressions of Mims (Phys. Rev. B. vol 5, p. 2409 (1972)
*     & vol 6, p. 3543 (1972) ).
*
*     tr:         EPR probabilities between alpha and beta manifold
*     trphi:      phase of transition moments
*     mult:       multiplicity of I-spin
*     chi0:       CHI0-term of modulation intensity
*     aa:         Basic ESEEM intensities of alpha manifold
*     ab:         Basic ESEEM intensities of beta manifold
*     ac:         Combination intensities ( CHIJKN )
*     index:      index array for combination intensities
*     nc:         number of combination intensities above threshold
*     maxc:       length of ac array
*                 IF maxc=0 THEN no combination intensities will be
*                 calculated
*----------------------------------------------------------------------
      IMPLICIT NONE
*-- arguments --
      INTEGER mult,nc,maxc,index(2,*)
      DOUBLE PRECISION tr(mult,mult),trphi(mult,mult),
     -   chi0,aa(*),ab(*),ac(*)
*-- local parameters --
      INTEGER ifreq,i,j,k,n
      DOUBLE PRECISION chij,chkn,chijkn,Salpha,Sbeta,thold
*
      INTEGER itrian,more,less
      DATA thold/1.0d-5/
      itrian(more,less)=((more-1)*(more-2))/2 + less
*            statement function to index lower trianular matrix
*            without diagonal
*
*>    CALCULATE BASIC MODULATION INTENSITIES
*
* - DC TERM ------------------
      chi0 = 0.0D0
      DO 100 i = 1 , mult
      DO 100 k = 1 , mult
  100   chi0 = chi0 + tr(i,k)**2
      chi0 =  chi0 / mult
* - ALPHA FREQUENCIES --------
      Salpha = 0.0
      ifreq = 0
      DO 200 i = 2 , mult
      DO 200 j = 1 , i-1
        chij = 0.0D0
        DO 150 k = 1 , mult
  150     chij = chij + TR(i,k)*TR(j,k)
        chij = chij / mult
        Salpha = Salpha + chij
        ifreq = ifreq + 1
        aa(ifreq) = chij * 2.d0
*>>                         if(mask(1,ifreq)) aa(ifreq)=0.d0
  200 CONTINUE
      Salpha = Salpha * 2.0D0

* - BETA  FREQUENCIES --------
      Sbeta = 0.0
      ifreq = 0
      DO 300 k = 2 , mult
      DO 300 n = 1 , k-1
        chkn = 0.0D0
        DO 250 i = 1 , mult
  250     chkn = chkn + tr(i,k)*tr(i,n)
        chkn = chkn / mult
        Sbeta = Sbeta + chkn
        ifreq = ifreq + 1
        ab(ifreq) = chkn * 2.d0
*>>                          if(mask(2,ifreq)) ab(ifreq)=0.d0
  300 CONTINUE
      Sbeta  = Sbeta * 2.0D0
*
* -- COMBINATION FREQUENCIES
*
      if(maxc.lt.1) return
*
      ifreq = 0
*
      DO 500 i = 2 , mult
      DO 500 j = 1 , i-1
      DO 500 k = 2 , mult
      DO 500 n = 1 , k-1
*>>      if(mask(1,itrian(i,j)).or.mask(2,itrian(k,n)) goto 500
        chijkn = TR(i,k)*TR(i,n)*TR(j,n)*TR(j,k)
        chijkn = DSQRT(chijkn)*DCOS(TRPHI(i,n)+TRPHI(j,k)
     -                           -TRPHI(i,k)-TRPHI(j,n))
        IF(DABS(chijkn).GT.thold) THEN
          chijkn = chijkn * 2.0 / mult
          ifreq = ifreq + 1
          if(ifreq.gt.maxc) stop ' too many combination frequencies'
          ac(ifreq) = chijkn
          index(1,ifreq) = itrian(i,j)
          index(2,ifreq) = itrian(k,n)
        ENDIF
 500  CONTINUE
      nc = ifreq
      END

*-------------------------------------------------------------------
      SUBROUTINE protese(B,k,l,gn,A,fa,fb,a0,aa,ab,ac)
*-------------------------------------------------------------------
*     INPUT
*     generation of ESEEM modulation intensities for I=1/2 systems
*     B     = Magnetic field magnitude (gauss)
*     k     = orientation of magnetic field
*     l     = orientation of effective field (anisotropic electron g)
*     gn    = nuclear g-factor
*     A     = Hyperfine tensor (10-4 cm-1)
*     OUPUT
*     fa/fb = Hyperfine frequencies (alpha/beta)
*     a0    = DC modulation term
*     aa/ab = modulation amplitude for alpha/beta frequency
*     ac    = amplitude of combination frequency
*

      IMPLICIT none
      INTEGER XX,XY,XZ,YY,YZ,ZZ,X,Y,Z
      PARAMETER(XX=1,XY=2,YY=3,XZ=4,YZ=5,ZZ=6,X=1,Y=2,Z=3)
*---  arguments
      DOUBLE PRECISION B,k(3),l(3),gn,A(6),fa,fb,a0,aa,ab,ac
*---  local
      DOUBLE PRECISION HEPV(3),HEMV(3),
     .                 HEM,HEP,H0,HFV(3),HF,gnucmag,cosa,sina,kmod
*---  global
      DOUBLE PRECISION pi,planck,plinv,bohr,nucmag,boltzk
      COMMON /CONSTANTS/  pi,planck,plinv,bohr,nucmag,boltzk
*
      gnucmag = gn * nucmag
      H0 = b*gnucmag
*
      HFV(X) = L(X) * A(XX) + L(Y) * A(XY) + L(Z) * A(XZ)
      HFV(Y) = L(X) * A(XY) + L(Y) * A(YY) + L(Z) * A(YZ)
      HFV(Z) = L(X) * A(XZ) + L(Y) * A(YZ) + L(Z) * A(ZZ)
*---         Hyperfine field
      HF = DSQRT(HFV(X)**2+HFV(Y)**2+HFV(Z)**2)
      cosa = (k(x)*hfv(x)+k(y)*hfv(y)+k(z)*hfv(z))/HF
      sina = DSQRT(1-cosa**2)
*---         Effective field (10-4 cm-1)
      HEPV(X) = H0*K(X) + HFV(X)/ 2
      HEPV(Y) = H0*K(Y) + HFV(Y)/ 2
      HEPV(Z) = H0*K(Z) + HFV(Z)/ 2
      HEP = DSQRT(HEPV(X)**2+HEPV(Y)**2+HEPV(Z)**2)
      HEMV(X) = H0*K(X) - HFV(X)/ 2
      HEMV(Y) = H0*K(Y) - HFV(Y)/ 2
      HEMV(Z) = H0*K(Z) - HFV(Z)/ 2
      HEM = DSQRT(HEMV(X)**2+HEMV(Y)**2+HEMV(Z)**2)
*---         Frequencies
      fa = HEP * plinv
      fb = HEM * plinv
*---         Modulation intensities
      IF(HEP.LT.1.D-6.OR.HEM.LT.1.D-6) THEN
*--->>>  cancellation condition
         kmod = 1.D0
      ELSE
         kmod  = (H0*HF*sina)/(HEP*HEM)
         kmod  = kmod**2
      ENDIF
      a0 = 1 - kmod/2
      aa = kmod/2
      ab = kmod/2
      ac = -kmod/4
      END
*----------------------------------------------------------------------
      SUBROUTINE genfreq(EA,EB,fa,fb,mult)
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER mult,ifreq,i,j
      DOUBLE PRECISION ea(*),eb(*),fa(*),fb(*)
*
      DOUBLE PRECISION pi,planck,plinv,bohr,nucmag,boltzk
      COMMON /CONSTANTS/  pi,planck,plinv,bohr,nucmag,boltzk

      ifreq = 0
      DO 100 i = 2 , mult
      DO 100 j = 1 , i-1
        ifreq = ifreq + 1
        fa(ifreq) = DABS(EA(I) - EA(J)) * plinv
        fb(ifreq) = DABS(EB(I) - EB(J)) * plinv
  100 CONTINUE
      END

*----------------------------------------------------------------------
      SUBROUTINE ese2p(fa,fb,a0,aa,ab,ac,index,ntran,nc,atab,ftab,ntab)
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER index(2,*),ntran,nc,i,ifr,ntab
      DOUBLE PRECISION fa(*),fb(*),a0,aa(*),ab(*),ac(*),
     .                 atab(0:*),ftab(0:*)
      if(ntran.le.0) return
*.... zero frequency
      ftab(0) = 0.D0
      atab(0) = a0
*.... basic frequencies
      ifr = 0
      DO 100 i = 1 , ntran
             ifr = ifr + 1
             ftab(ifr) = fa(i)
             atab(ifr) = aa(i)
             ifr = ifr + 1
             ftab(ifr) = fb(i)
             atab(ifr) = ab(i)
 100  CONTINUE
      ntab = ifr+1
*
*.... combination frequencies
      if(nc.lt.1) return
      DO 200 i = 1 , nc
             ifr = ifr + 1
             ftab(ifr) = fa(index(1,i)) + fb(index(2,i))
             atab(ifr) = ac(i)
             ifr = ifr + 1
             ftab(ifr) = dabs(fa(index(1,i)) - fb(index(2,i)) )
             atab(ifr) = ac(i)
 200  CONTINUE
      ntab = ifr+1
      END

*----------------------------------------------------------------------
      SUBROUTINE ese3psup(fa,fb,a0a,a0b,aa,ab,ac,index,ntran,nc,tau)
*----------------------------------------------------------------------
*.... three pulse eseem suppression effect
*
      IMPLICIT none
      INTEGER index(2,*),ntran,nc,i,ij,kn
      DOUBLE PRECISION fa(*),fb(*),a0a,a0b,aa(*),ab(*),ac(*),tau,pi
      DATA pi /3.141592653589793D0/
      if(ntran.le.0) return
*
*.... suppression effect
      if(tau.eq.0.D0) return

      DO 10 i = 1 , ntran
      a0a = a0a +  ab(i)*dcos(2*pi*fb(i)*tau)
  10  a0b = a0b +  aa(i)*dcos(2*pi*fa(i)*tau)

      DO 100 i = 1 , nc
         ij = index(1,i)
         kn = index(2,i)
         aa(ij)=aa(ij)+ac(i)*2*dcos(2*pi*fb(kn)*tau)
         ab(kn)=ab(kn)+ac(i)*2*dcos(2*pi*fa(ij)*tau)
 100  CONTINUE
      END
