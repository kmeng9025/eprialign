*---------------------------------------------------------------------
*
      SUBROUTINE SETEPR(nwords,ip,maclevel,ircode)
*---------------------------------------------------------------------
      IMPLICIT none
      INTEGER nkeys,ipos,ip,nwords,index,noccur,maclevel,ircode,mxval
      PARAMETER(nkeys=8,mxval=20)
      LOGICAL ok
      CHARACTER*10 menu(nkeys)
      CHARACTER*8  typmenu(3)
      CHARACTER*60 help(nkeys)
*---> global
      INTEGER merror
      COMMON /macro/  merror
*---> local
      DOUBLE PRECISION rval(mxval)
      INTEGER ival(mxval),vindexR(mxval),vindexI(mxval),narg
      DATA MENU /'G-TENSOR  ','SPLITTING ','RESET     ','XTYPE     ',
     .           'EXCITATION','EXP       ','LIST      ','HELP      '/
      DATA TYPMENU /'NONE    ','GAUSS   ','LORENTZ '/
      DATA HELP
     ./' syntax: G(1) G(2) G(3) Alpha Beta Gamma                    ',
     . ' syntax: index mult A(1) A(2) A(3) Alpha Beta Gamma         ',
     . ' syntax:                                                    ',
     . ' syntax: type [NONE/GAUSS/LORENTZ]                          ',
     . ' syntax: xfreq xwidth                                       ',
     . ' syntax: krid npoints lowfield highfield                    ',
     . ' syntax:                                                    ',
     . ' Produces this text                                         '/
      ipos = ip
      ircode = 0
    1 continue
      ok = .true.
      goto(2,3,3) maclevel
    2 continue
*----> interpretation/definition level:
        IF(ip.eq.0) THEN
           call message(ircode)
           !write(6,*) ' SETEPR>'
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
      goto(10,20,30,40,50,60,70,80)index
      !write(6,*) ' illegal code SETEPR',index
      stop
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*--->  G-TENSOR  (G(1),G(2),G(3),alpha,beta,gamma)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   10 continue
      call macparse(nwords,ipos+1,maclevel,index,0,6,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefR(rval,vindexR,6)
      IF(maclevel.eq.0.or.maclevel.eq.3) call eprG(rval)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> SPLITTING index mult A(1),A(2),A(3), alpha,beta,gamma
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   20 continue
      call macparse(nwords,ipos+1,maclevel,index,2,6,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,2)
      call mderefR(rval,vindexR,6)
      IF(maclevel.eq.0.or.maclevel.eq.3) call eprA(ival,rval)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> RESET
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   30 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call eprreset
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> XTYPE NONE/GAUSS/LORENTZ
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   40 continue
      CALL recog(ipos+1,typmenu,3,index,noccur)
      if(noccur.ne.1)goto 1313
      call eprtype(index-1)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> EXCITATION  freq width
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   50 continue
      call macparse(nwords,ipos+1,maclevel,index,0,2,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefR(rval,vindexR,2)
      IF(maclevel.eq.0.or.maclevel.eq.3) call eprexc(rval)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> EXP ( krid npoints lowfield highfield )
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   60 continue
      call macparse(nwords,ipos+1,maclevel,index,2,2,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.lt.0) goto 1313
      if(narg.gt.0) then
        call mderefI(ival,vindexI,2)
        call mderefR(rval,vindexR,2)
        IF(maclevel.eq.0.or.maclevel.eq.3) call eprexp(ival,rval)
      else
        call listeprexp
        goto 1515
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> LIST
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   70 continue
      call eprlist
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> help list
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   80 continue
      call helplst(menu,help,nkeys)
      goto 1515
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

***********************************************************************
      SUBROUTINE eprreset
*----------------------------------------------------------------------
* initialize EPR1 experimental module COMMON blocks
*
      IMPLICIT none
***v
      INTEGER nCnuc,cmult,Xtype,maxnuc
      PARAMETER (maxnuc=10)
      DOUBLE PRECISION AE,GG,AT,GT,Xwidth,Xfreq
      COMMON /corR8/ AE(6,maxnuc),AT(6,maxnuc),GG(6),GT(6),Xfreq,Xwidth
      COMMON /corI4/ nCnuc,cmult(maxnuc),Xtype
***^
*     local
      INTEGER i,j,maxval,index,inuc
      PARAMETER(maxval=6)
      DOUBLE PRECISION Rval(maxval),R(3,3)
      INTEGER Ival(maxval)
      CALL dzero(AE,6*maxnuc)
      CALL dzero(AT,6*maxnuc)
      CALL dzero(GG,6)
      CALL dzero(GT,6)
      CALL izero(cmult,maxnuc)
      xtype = 0
      ncnuc = 0
      xwidth = 100D0
      xfreq  = 9000D0
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY eprlist
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      !write(6,'(A,6(1X,G10.5))') ' GG: ' ,(GG(i),i = 1 , 6)
      !write(6,'(A,6(1X,G10.5))') ' GT: ' ,(GT(i),i = 1 , 6)
      DO 10 j = 1 , ncnuc
      !write(6,'(A,I1,6(1X,G11.4))') ' AE',j,(AE(i,j),i = 1 , 6)
      !write(6,'(A,I1,6(1X,G11.4))') ' AT',j,(AT(i,j),i = 1 , 6)
   10 continue
      !write(6,*) ' Xtype=',xtype
      !write(6,*) ' Xfreq,width=',xfreq,xwidth
      return
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY EPRG(Rval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      DO 20 i = 1 , 6
   20 GG(i) = Rval(i)
      call euler2cos(GG(4),R)
      call Trot(GG,R,GT)
      return
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY EPRA(Ival,Rval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if(ival(1).le.0.or.ival(1).ge.ncnuc+1) then
         ncnuc = ncnuc+1
         inuc  = ncnuc
      else
         inuc  = ival(1)
      endif
      if(ncnuc.gt.maxnuc) stop ' eprA: too many nuclei'
      cmult(inuc) = Ival(2)
      DO 30 i = 1 , 6
   30 AE(i,inuc) = Rval(i)
      call euler2cos(AE(4,inuc),R)
      call Trot(AE(1,inuc),R,AT(1,inuc))
      return
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY EPRtype(index)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      xtype = index
      return
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY EPRexc(Rval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      xfreq = rval(1)
      xwidth = rval(2)
      return
      END

*----------------------------------------------------------------------
*     INTERFACING SECTION
*----------------------------------------------------------------------
      SUBROUTINE epr(xmin,xmax,ifmt,accu,n,max,npag)
*----------------------------------------------------------------------
*     INTERFACE TO data-page-driver
*     accu: real*8 accumulation buffer
*     max:  n1-dimension of page-system
*     npag: number of pages available
*     output:
*     n:    number of points in buffer
*     xmin/xmax x-axis limits
*     ifmt: format(-8:double/8:complex/4:real)
*---------------------------------------
      IMPLICIT none
      INTEGER n(*),max,ifmt(*),npag
      REAL xmin(*),xmax(*)
      DOUBLE PRECISION accu(max,*)
*-->local
      INTEGER krid,nac,maxval
      PARAMETER (maxval=5)
      INTEGER ival(maxval)
      DOUBLE PRECISION Blo,Bhi,Rval(maxval)
      DATA Blo/2000.0/,Bhi/4000.0/,krid/5/,nac/512/
*
      if(nac.le.0.or.nac.gt.max)
     .  stop ' illegal argument in EPR'
      call simepr1(Blo,Bhi,krid,accu,nac)
      CALL r8r4(accu,accu,nac)
      xmin(1) = Blo
      xmax(1) = Bhi
      ifmt(1) = 4
      n(1)    = nac
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY eprexp(ival,rval)
*----> EXP ( krid npoints lowfield highfield )
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      krid = ival(1)
      nac  = ival(2)
      Blo  = Rval(1)
      Bhi  = Rval(2)
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY listeprexp
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      !write(6,'('' KRID ='',I10  ,'' NAC='',I10)') krid,nac
      !write(6,'('' Blo='',G10.3,'' Bhi='',G10.3)')Blo,Bhi
      END

*----------------------------------------------------------------
      SUBROUTINE simepr1(Blo,Bhi,krid,accu,nac)
*----------------------------------------------------------------
*     KEY-routine for 1-st order EPR spectrum
*     CALLED BY: simese
*----------------------
      IMPLICIT none
***v
      INTEGER nCnuc,cmult,Xtype,maxnuc
      PARAMETER (maxnuc=10)
      DOUBLE PRECISION AE,GG,AT,GT,Xwidth,Xfreq
      COMMON /corR8/ AE(6,maxnuc),AT(6,maxnuc),GG(6),GT(6),Xfreq,Xwidth
      COMMON /corI4/ nCnuc,cmult(maxnuc),Xtype
***^
*     arguments
      INTEGER krid,nac
      DOUBLE PRECISION Blo,Bhi,accu(nac)
*     local
      INTEGER ncalls,IS(maxnuc),MIS(maxnuc),i,ib
      DOUBLE PRECISION theta,phi,th,ph,K(3),L(3),GE,Cfreq,B,BR,
     .                 T(maxnuc),TT
      LOGICAL xlast,xdone
      DOUBLE PRECISION BOHR,PLANCK,pi
      DATA bohr   /0.46688204D0/
*          Bohr-magneton in 10-4 cm-1 / gauss
      DATA pi     /3.141592653589793D0/
      DATA planck /0.3335645D0/
*          Planck constant in 10-4 cm-1 / MHz

      B = (Blo+Bhi)/2.D0
*
*---  POWDER INTEGRATION
      ncalls = 0
  10  CONTINUE
      call pwdpeal(krid,theta,phi,ncalls,xlast)
      th = theta*pi/180.D0
      ph = phi  *pi/180.D0
      k(1)=DSIN(th)*DCOS(ph)
      k(2)=DSIN(th)*DSIN(ph)
      k(3)=DCOS(th)
*
      call geff(GT,B,K,L,GE,Cfreq)
      DO 15 i = 1 , nCnuc
   15 call Aeff(L,AT(1,i),T(i))
      DO 20 i = 1 , nCnuc
         IS(i) = cmult(i) - 1
         MIS(i)= IS(i)
         Cfreq = Cfreq + MIS(i)*T(i)/(2.D0*planck)
   20 CONTINUE
*
   30 CONTINUE
*       resonant field BR:
        BR = B + (Xfreq-Cfreq)*planck/(GE*BOHR)
        ib = idnint((BR-Blo)*DBLE(nac)/(Bhi-Blo))
        if(ib.ge.1.and.ib.le.nac) accu(ib) = accu(ib) + 1.D0
        call nextmm(IS,MIS,T,TT,ncnuc,xdone)
        Cfreq = Cfreq + TT/planck
      if(.not.xdone) goto 30
*
      if(.not.xlast) goto 10
      END

*----------------------------------------------------------------
      SUBROUTINE EPRSELECT(B,K,L,factor)
*----------------------------------------------------------------
*     KEY-routine for orientation selection
*     CALLED BY: simese
*----------------------
      IMPLICIT none
***v
      INTEGER nCnuc,cmult,Xtype,maxnuc
      PARAMETER (maxnuc=10)
      DOUBLE PRECISION AE,GG,AT,GT,Xwidth,Xfreq
      COMMON /corR8/ AE(6,maxnuc),AT(6,maxnuc),GG(6),GT(6),Xfreq,Xwidth
      COMMON /corI4/ nCnuc,cmult(maxnuc),Xtype
***^
*     arguments:
      DOUBLE PRECISION B,K(3),L(3),factor
*     local :
      DOUBLE PRECISION T(maxnuc),Cfreq,GE
      INTEGER i
*
      IF(xtype.le.0.or.xtype.gt.2) then
         factor = 1.D0
         L(1) = K(1)
         L(2) = K(2)
         L(3) = K(3)
         return
      ENDIF
*
      call geff(GT,B,K,L,GE,Cfreq)
      DO 15 i = 1 , nCnuc
  15  call Aeff(L,AT(1,i),T(i))
      call eprweigh(Cfreq,T,cmult,nCnuc,Xtype,Xfreq,Xwidth,factor)
      END


*-----------------------------------------------------------------------
      SUBROUTINE eprweigh(Cfreq,T,mult,npart,Xtype,Xfreq,Xwidth,factor)
*-----------------------------------------------------------------------
      IMPLICIT none
      INTEGER npart, mult(*),mxpart,Xtype,i
      PARAMETER(mxpart=10)
      DOUBLE PRECISION Cfreq,T(*),Xfreq,Xwidth,factor
*     local
      DOUBLE PRECISION eprfreq,plinv,TT
      INTEGER IS(mxpart),MIS(mxpart)
      LOGICAL xdone
* 1/h:(Planck's constant in MHZ / 10-4cm-1)
      DATA plinv  /2.9979209D0/
*
      if(npart.gt.mxpart) stop ' !EPRWEIGH: -- to many particles'
      DO 10 i = 1 , npart
         IS(i) = mult(i) - 1
         MIS(i)= IS(i)
   10 CONTINUE
      factor = 0.D0
      eprfreq = Cfreq
      DO 30 i = 1 , npart
   30 eprfreq = eprfreq + MIS(i)*T(i)*plinv/2.D0
*
   20 CONTINUE
      call xweigh(factor,eprfreq,Xtype,Xfreq,Xwidth)
      call nextmm(IS,MIS,T,TT,npart,xdone)
      eprfreq = eprfreq + TT*plinv
      if(.not.xdone) goto 20
*
      END

*-----------------------------------------------------------------------
      SUBROUTINE nextmm (IS,MIS,T,TT,npart,xdone)
*-----------------------------------------------------------------------
*     compute next set of quantum numbers of npart particles
*     IS:   I or S quantum number (*2)
*     MIS:  Mi or Ms quantum number (*2)
*
      IMPLICIT none
      LOGICAL xdone
      INTEGER npart,IS(*),MIS(*),k
      DOUBLE PRECISION T(*), TT
      k = NPART
      xdone = .FALSE.
      TT = 0.D0
      if(k.eq.0) then
         xdone = .true.
         return
      endif
  100 CONTINUE
      IF (MIS(k).GT.-IS(k)) THEN
        MIS(k) = MIS(k) - 2
        TT = TT -T(k)
        RETURN
      ELSE IF (k.EQ.1) THEN
        xdone = .TRUE.
        RETURN
      ELSE
        MIS(k) = IS(k)
        TT = TT + MIS(k)*T(k)
        k = k - 1
      ENDIF
      GOTO 100
      END
*----------------------------------------------------------------
      SUBROUTINE xweigh(factor,freq,Xtype,Xfreq,Xwidth)
*----------------------------------------------------------------
      IMPLICIT none
      INTEGER Xtype
      DOUBLE PRECISION factor,freq,Xfreq,Xwidth,
     .                 gauss,lorentz,lwfrac,x,width,deltaf
*
      gauss(lwfrac) = DEXP(-2.7725887224 * lwfrac**2)
      lorentz(x,width) = (width/2.0)**2/((width/2.0)**2 + x**2)
*                                        gauss(1.5) = 0.002..
*                                      lorentz(1,11)= 0.002..
      deltaf = dabs(freq-xfreq)
      goto (10,20) xtype
      stop ' shapex -- illegal xtype'
   10 CONTINUE
      if(deltaf/xwidth.gt.1.5) return
      factor = factor + gauss(deltaf/xwidth)
      return
*
   20 CONTINUE
      if(deltaf/xwidth.gt.11) return
      factor = factor + lorentz(deltaf,xwidth)
      END
*----------------------------------------------------------------
      SUBROUTINE Geff(G,B,K,L,GE,FREQ)
*----------------------------------------------------------------
*---> compute effective field vector (l), G-effective and Frequency
      IMPLICIT NONE
      INTEGER XX,XY,XZ,YY,YZ,ZZ,X,Y,Z
      PARAMETER(XX=1,XY=2,YY=3,XZ=4,YZ=5,ZZ=6,X=1,Y=2,Z=3)
      DOUBLE PRECISION G(6),B,K(3),L(3),GE,FREQ,Bohr,plinv
      DATA bohr   /0.46688204D0/
      DATA plinv  /2.9979209D0/
*
      L(X) = K(X) * G(XX) + K(Y) * G(XY) + K(Z) * G(XZ)
      L(Y) = K(X) * G(XY) + K(Y) * G(YY) + K(Z) * G(YZ)
      L(Z) = K(X) * G(XZ) + K(Y) * G(YZ) + K(Z) * G(ZZ)
*
      GE = DSQRT(L(X)**2 + L(Y)**2 + L(Z)**2)
*
      L(X) = L(X)/GE
      L(Y) = L(Y)/GE
      L(Z) = L(Z)/GE
*
      FREQ = BOHR * B * GE * plinv
*
      END

*----------------------------------------------------------------
      SUBROUTINE Aeff(L,A,T)
*----------------------------------------------------------------
*---> compute effective HYPERFINE (T): strong field approximation
      IMPLICIT NONE
      INTEGER XX,XY,XZ,YY,YZ,ZZ,X,Y,Z
      PARAMETER(XX=1,XY=2,YY=3,XZ=4,YZ=5,ZZ=6,X=1,Y=2,Z=3)
      DOUBLE PRECISION A(6),L(3),LA(3),T
*
      LA(X) = L(X) * A(XX) + L(Y) * A(XY) + L(Z) * A(XZ)
      LA(Y) = L(X) * A(XY) + L(Y) * A(YY) + L(Z) * A(YZ)
      LA(Z) = L(X) * A(XZ) + L(Y) * A(YZ) + L(Z) * A(ZZ)
*
      T = DSQRT(LA(X)**2 + LA(Y)**2 + LA(Z)**2)
*
      END

      SUBROUTINE SELECT
      IMPLICIT none
      INTEGER idim1,idim2,ncalls,i,j,krid
      PARAMETER (idim1=10,idim2=20)
      DOUBLE PRECISION angles(0:idim1,0:idim2),theta,phi,th,ph,facmax
      DOUBLE PRECISION K(3),L(3),pi,factor,B
      LOGICAL xlast
      DATA pi     /3.141592653589793D0/
*
      DO 10 j = 0 , idim2
      DO 10 i = 0 , idim1
   10 angles(i,j) = 0.D0
*
      krid = 10
      facmax = 0D0
      ncalls= 0
      B = 3000D0
   20 continue
      call pwdpeal(krid,theta,phi,ncalls,xlast)
      th = theta*pi/180.D0
      ph = phi  *pi/180.D0
      k(1)=DSIN(th)*DCOS(ph)
      k(2)=DSIN(th)*DSIN(ph)
      k(3)=DCOS(th)
      call EPRSELECT(B,K,L,factor)
      i = idnint(theta*dble(idim1)/90D0)
      j = idnint(phi* dble(idim2)/360D0)
      angles(i,j) = angles(i,j) + factor
      if(facmax.lt.angles(i,j)) facmax = angles(i,j)
      if(.not.xlast) goto 20
      !write(6,*) 'angle density plot:'
      if(facmax.le.1.D-5) facmax = 1.0
      DO 30 i = 0 , idim1
      !write(6,'(80I1)') (idnint(angles(i,j)*10D0/facmax),j=0,idim2)
   30 continue
      END


      SUBROUTINE EPRTEST
      IMPLICIT none
      INTEGER nac,krid,i
      PARAMETER (nac=256)
      DOUBLE PRECISION accu(nac),Blo,Bhi
      Blo = 2700D0
      Bhi = 3700D0
      krid = 30
      DO 1 i = 1 , nac
    1 accu(i) = 0D0
      CALL simepr1(Blo,Bhi,krid,accu,nac)
      DO 10 i = 1 , nac
      !write(6,'(2(1X,G12.6))') Blo+dble(i)*(Bhi-Blo)/dble(nac),accu(i)
  10  continue
      END
