*---------------------------------------------------------------------
      PROGRAM testmr2
*---------------------------------------------------------------------
      IMPLICIT none
      INTEGER nkeys,nwords,index,noccur,ircode
      PARAMETER(nkeys=10)
      LOGICAL ok
      CHARACTER*10 menu(nkeys)
      CHARACTER*60 help(nkeys)
*--->local
      DOUBLE PRECISION GN,A(6),Q(6)
      DATA gn,a,q /13*0.D0/
      DATA MENU /'GN        ','AE        ','AT        ','AE        ',
     .           'QT        ','TAU       ','FIELD     ','CALC      ',
     .           'HELP      ','EXIT      '/
      DATA HELP
     ./' syntax: Nuclear G-value                                    ',
     . ' syntax: A(1) A(2) A(3) Alpha Beta Gamma                    ',
     . ' syntax: Axx Axy Ayy Axz Ayz Azz                            ',
     . ' syntax: Q(1) Q(2) Q(3) Alpha Beta Gamma                    ',
     . ' syntax: Qxx Qxy Qyy Qxz Qyz Qzz                            ',
     . ' syntax: tau-value (0:sup-free; +:3-pulse -:2-pulse )       ',
     . ' syntax: field-value + polar angles (theta,phi)             ',
     . '         (clear tensor info)                                ',
     . ' Produces this text                                         ',
     . ' (Exit)                                                     '/
      ircode = 0
    1 continue
        call recog(ipos,menu,nkeys,index,noccur)
        if(noccur.ne.1) goto 1313
    9 continue
*=======>>> COMMAND DISPATCHER <<<<<===========
      goto(10,20,30,40,50,60,70,80,90,100)index
      !write(6,*) ' illegal code in testmr2',index
      stop
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> GN enter nuclear g-factor
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   10 continue
      call gtreal(2,gn,ok)
      if(.not.ok) goto 1313
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> A-TENSOR (A1,A2,A3,alpha,beta,gamma)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   20 continue
      DO 21 i = 1 , 6
         call gtreal(i+1,AE(i),ok)
         if(.not.ok) goto 1313
   21 continue
      call euler2cos(AE(4),R).........
      !write(6,*) ' rotation matrix:'
      !write(6,'(3(1X,G12.6))') ((R(i,j), j=1,3),i=1,3)
      call Trot(AE.....AT..)
      !write(6,*) ' transformed matrix:'
      !write(6,'(6(1X,G12.6))') (AT(i), i = 1 , 6)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> A-TENSOR Axx Axy Ayy Axz Ayz Azz
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   30 continue
      DO 31 i = 1 , 6
         call gtreal(i+1,AT(i),ok)
         if(.not.ok) goto 1313
   31 continue
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> Q-TENSOR (Q1,Q2,Q3,alpha,beta,gamma)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   40 continue
      DO 41 i = 1 , 6
         call gtreal(i+1,QE(i),ok)
         if(.not.ok) goto 1313
   41 continue
      call euler2cos(QE(4),R).........
      !write(6,*) ' rotation matrix:'
      !write(6,'(3(1X,G12.6))') ((R(i,j), j=1,3),i=1,3)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> Q-TENSOR Qxx Qxy Qyy Qxz Qyz Qzz
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   50 continue
      DO 51 i = 1 , 6
         call gtreal(i+1,QT(i),ok)
         if(.not.ok) goto 1313
   51 continue
      call hdiag(QT,QE,R) ............
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> help TAU
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   60 continue
      call gtreal(2,tau,ok)
      if(.not.ok) goto 1313
      goto 4711 
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> FIELD
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   70 continue
      call gtreal(2,B,ok)
      if(.not.ok) goto 1313
      call gtreal(3,theta,ok)
      if(.not.ok) goto 1313
      call gtreal(4,phi,ok)
      if(.not.ok) goto 1313
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> CALC
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   80 continue
      th = theta*pi/180.D0
      ph = phi  *pi/180.D0
      k(1)=DSIN(th)*DCOS(ph)
      k(2)=DSIN(th)*DSIN(ph)
      k(3)=DCOS(th)
      BX=B*K(1)
      BY=B*K(2)
      BZ=B*K(3)
      LX=bx
      LY=by
      LZ=bz
      CALL hypfield(bx,by,bz,lx,ly,lz,gn,A,hx,hy,hz)
      CALL spin1(hx(1),hy(1),hz(1),Q,R,EA,Vr(1,1),Vi(1,1))
      CALL spin1(hx(2),hy(2),hz(2),Q,R,EB,Vr(1,2),Vi(1,2))
      CALL inprod(Vr(1,1),Vi(1,1),Vr(1,2),Vi(1,2),Rprob,Iprob,mult)
      CALL complx(Rprob,Iprob,Mprob,Fprob,mult*mult,1)
      CALL mims23(Mprob,Fprob,mult,a0,aa,ab,ac,index,nc,maxc)
      CALL genfreq(EA,EB,fa,fb,mult)
      IF (tau.lt.0.d0) THEN
          CALL ese2p(fa,fb,a0,aa,ab,ac,index,ntran,nc,atab,ftab,ntab)
          if(idebug.eq.1) !write(6,*) ' ntran, nc, ntab',ntran,nc,ntab
          CALL nuccum(atab,ftab,ntab,1,x2(i),i,nnuc,accu,nac,factor)
      ELSE
          a0a = a0
          a0b = a0
          CALL ese3psup(fa,fb,a0a,a0b,aa,ab,ac,index,ntran,nc,tau)
      ENDIF
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> help list
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  90  continue
      call helplst(menu,help,nkeys)
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> EXIT
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  100 continue
*---> TERMINATION
 4711 ircode = 0
      goto 1
 1313 continue
      ircode = -1
      !write(6,*) ' INPUT not recognized'
      goto 1
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
