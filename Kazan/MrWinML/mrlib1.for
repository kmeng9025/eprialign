* = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
      BLOCK DATA
* = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
      IMPLICIT none
*-----                  /GLOBAL CONSTANTS/
*
      DOUBLE PRECISION pi,planck,plinv,bohr,nucmag,boltzk
      COMMON /CONSTANTS/  pi,planck,plinv,bohr,nucmag,boltzk
      DATA bohr   /0.46688204D0/
*          Bohr-magneton in 10-4 cm-1 / gauss
      DATA pi     /3.141592653589793D0/
      DATA planck /0.3335645D0/
*          Planck constant in 10-4 cm-1 / MHz
      DATA plinv  /2.9979209D0/
      DATA nucmag /0.0002542775392D0/
*          proton magneton in 10-4 cm-1 /gauss
      DATA boltzk /6950.239312D0/
*          boltzman constant in 10-4 cm-1 / K
*     DATA E/8.0657D3/,G/2.002319D0/,BETA/4.668598D-5/,
*    .PLANCK/3.335640D-11/,HSTR/5.3088366D-12/,PI/3.141592653589793D0/

      END

*----------------------------------------------------------------
      SUBROUTINE SPINHALF(G,GLEN,B,KX,KY,KZ,LX,LY,LZ,GEFF,FREQ)
*----------------------------------------------------------------
*
      IMPLICIT NONE
      INTEGER GLEN,XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ
      PARAMETER(XX=1,XY=2,YY=3,XZ=4,YZ=5,ZZ=6)
      DOUBLE PRECISION G(6),B,KX,KY,KZ,LX,LY,LZ,GEFF,FREQ
      DOUBLE PRECISION pi,planck,plinv,bohr,nucmag,boltzk
      COMMON /CONSTANTS/  pi,planck,plinv,bohr,nucmag,boltzk
*
      IF(GLEN.EQ.1) THEN
         LX = KX
         LY = KY
         LZ = KZ
         GEFF = G(1)
      ELSE IF(GLEN.EQ.6) THEN
         LX = KX * G(XX) + KY * G(XY) + KZ * G(XZ)
         LY = KX * G(XY) + KY * G(YY) + KZ * G(YZ)
         LZ = KX * G(XZ) + KY * G(YZ) + KZ * G(ZZ)
*
         GEFF = DSQRT(LX*LX + LY*LY + LZ*LZ)
*
         LX = LX/GEFF
         LY = LY/GEFF
         LZ = LZ/GEFF
      ELSE
         STOP ' SPINHALF ILLEGAL GLEN'
      ENDIF
*
      FREQ = BOHR * B * GEFF / PLANCK
*
      END

*---------------------------------------------------------------
      SUBROUTINE HYPFIELD(BX,BY,BZ,LX,LY,LZ,G,A,HX,HY,HZ)
*---------------------------------------------------------------
      IMPLICIT NONE
      INTEGER XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ
      PARAMETER(XX=1,XY=2,YY=3,XZ=4,YZ=5,ZZ=6)
      DOUBLE PRECISION A(6),LX,LY,LZ,G,HX(2),HY(2),HZ(2),
     .                      BX,BY,BZ,HFX,HFY,HFZ,GNUCMAG
      DOUBLE PRECISION pi,planck,plinv,bohr,nucmag,boltzk
      COMMON /CONSTANTS/  pi,planck,plinv,bohr,nucmag,boltzk
*
      GNUCMAG = G * NUCMAG
*
      HFX = LX * A(XX) + LY * A(XY) + LZ * A(XZ)
      HFY = LX * A(XY) + LY * A(YY) + LZ * A(YZ)
      HFZ = LX * A(XZ) + LY * A(YZ) + LZ * A(ZZ)
*
      HX(1) = GNUCMAG * BX + HFX/ 2
      HX(2) = GNUCMAG * BX - HFX/ 2
      HY(1) = GNUCMAG * BY + HFY/ 2
      HY(2) = GNUCMAG * BY - HFY/ 2
      HZ(1) = GNUCMAG * BZ + HFZ/ 2
      HZ(2) = GNUCMAG * BZ - HFZ/ 2
      END

*---------------------------------------------------------------
      SUBROUTINE SPIN1(HX,HY,HZ,QQ,R,E,VR,VI)
*---------------------------------------------------------------
*
*     ANALYTICAL EVALUATION OF EIGENVALUES AND VECTORS OF NUCLEAR
*     SPIN = 1 HAMILTONIAN WITH ISOTROPIC G-VALUE AND ARBITRARY
*     QUADRUPOLE INTERACTION.
*
*     ARGUMENTS:
*     HXYZ     : COMPONENTS OF EFFECTIVE ZEEMAN FIELD (in 10-4 cm-1)
*     QQ(3)    : QUADRUPOLE PRINCIPAL VALUES
*     R(9)     : DIRECTION COSINES OF QUADRUPOLE TENSOR
*     E(3)     : EIGENVALUES
*     VR(3,3)  : REAL PART OF EIGENVECTORS
*     VI(3,3)  : IMAG PART OF EIGENVECTORS
*
*     ANALYTICAL EXPRESSIONS ACCORDING TO G.M MUHA J. MAGN. RESON.
*                                                    49,431 (1982)
*
*-------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ,X,Y,Z
      PARAMETER(XX=1,XY=2,XZ=3,YX=4,YY=5,YZ=6,ZX=7,ZY=8,ZZ=9)
      PARAMETER(X=1,Y=2,Z=3)
      DOUBLE PRECISION pi,planck,plinv,bohr,nucmag,boltzk
      COMMON /CONSTANTS/  pi,planck,plinv,bohr,nucmag,boltzk
      DOUBLE PRECISION HX,HY,HZ,QQ(3),R(9),E(3),VR(3,3),VI(3,3),
     -                 D2,DX,DY,DZ,P,Q,PABS3,BETA3,C,H,RPLUS,RMINUS
      INTEGER*4 K
*
*     ZEEMAN INTERACTION
*
*
*     CALCULATION OF DX,DY,DZ IN PRINCIPLE AXIS SYSTEM OF QQ
*
      DX =  HX * R(XX) + HY * R(YX) + HZ * R(ZX)
      DY =  HX * R(XY) + HY * R(YY) + HZ * R(ZY)
      DZ =  HX * R(XZ) + HY * R(YZ) + HZ * R(ZZ)
*
      d2 = dx*dx + dy*dy + dz*dz
*
*     CALCULATION OF P,Q,BETA,C PARAMETERS
*
      P = QQ(X)*QQ(Y) + QQ(X)*QQ(Z) + QQ(Y)*QQ(Z) - D2
      Q = QQ(X)*QQ(Y)*QQ(Z) - QQ(X)*DX*DX - QQ(Y)*DY*DY - QQ(Z)*DZ*DZ
      PABS3 = DABS(P)/3.0
      C = Q / (PABS3 * DSQRT(PABS3) * 2.0)
      BETA3 = DACOS(C)/3.0
      H = DSQRT(4*PABS3)
*
*     CALCULATION OF EIGENVALUES
*
      E(1) = H * DCOS(BETA3)
      E(2) = H * DCOS(BETA3 + 2.0*PI/3.0)
      E(3) = H * DCOS(BETA3 + 4.0*PI/3.0)
*
*     CALCULATIONS OF EIGENVECTORS
*
* -->    NUMERICAL STABILITY
      IF(DABS(DX).LT.1.0D-8 ) DX = 1.0D-8
      IF(DABS(DY).LT.1.0D-8 ) DY = 1.0D-8
      IF(DABS(DZ).LT.1.0D-8 ) DZ = 1.0D-8
*
      DO 50 K = 1,3
        RPLUS  = QQ(Y) - E(K)
        RMINUS = QQ(X) - E(K)
        IF(DABS(RPLUS) .LT.1.0D-5 )  RPLUS = 0.0
        IF(DABS(RMINUS).LT.1.0D-5 ) RMINUS = 0.0
*
        VR(1,K) = -DX * RMINUS
        VI(1,K) = DY * DZ
        VR(2,K) = RPLUS * RMINUS - DZ * DZ
        VI(2,K) = 0.0D0
        VR(3,K) = DZ * DX
        VI(3,K) = -DY * RPLUS
   50 CONTINUE
*
      CALL VNORM(VR,VI,3)
*
      END
*----------------------------------------------------------------------
***>>>ROUTINES FOR GENERAL SPIN SYSTEM
*----------------------------------------------------------------------
      SUBROUTINE spinZF(Sx,Sz,SSxy,SzS,Q,mult,mxmul,nnuc)
*----------------------------------------------------------------------
*     THIS ROUTINE PREPARES THE COMPONENTS OF THE SPIN MATRICES
*     NEEDED FOR SETTING UP A SPINHAMILTONIAN FOR A SINGLE SPIN IN A
*     ZEEMAN FIELD AND A ZERO FIELD INTERACTION OR NQI (K,eta)
*     IN THE PRINCIPAL AXIS SYSTEM OF THE ZFI/NQI
*     INSIDE LOOP OVER ORIENTATIONS, SUBROUTINE spinZE(....) SHOULD BE
*     CALLED TO INCLUDE THE ZEEMAN INTERACTION AND SETUP THE H-MATRIX
*
*     H = K((3Sz^2-S^2) + eta(Sx^2-Sy^2)) + DxSx+DySy+DzSz
*
*     Sz(mult)     : represents the diagonal of Sz spin matrix
*     Sx(mult-1)   : represents the sub diagonal of the Sx matrix
*                    the Sy can be easily evaluated from this
*     SSxy(mult-2) : represents the sub2 diagonal of (Sx^2-Sy^2)*K*eta
*     SzS(mult)    : represents the diagonal (2Sz^2-S^2) * K
*
*     Q(3)         : principal values of NQI/ZFI (10-4 cm-1)
*
*     Used theory:
*     Sx = (S+ + S-)/2; Sy = (S+ - S-)/2i
*     Sx^2-Sy^2 = (S+^2+S-^2)/2
*     S-|S,m> = sqrt((S-m+1)(S+m))|S,m-1>
*     S+|S,m> = sqrt((S+m+1)(S-m))|S,m-1>
*
*     NOTE that S-m+1 indexes the Sz eigenstates starting from the
*          maximum Ms-value.
*          For example: S=1, m=1 => S-m+1 = 1
*                       S=1, m=0 => S-m+1 = 2
*                       S=1, m=-1=> S-m+1 = 3
*
*   LET k represent the index of the eigenstate of Sz [k=S-m+1]
*   S = (mult(i)-1)/2 ==> m = FLOAT(mult(i)-2*k+1)/2
*              (S-m+1)(S+m) = FLOAT(k*(mult(i)-k)
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER nnuc,mxmul,mult(nnuc),i,k
      DOUBLE PRECISION Sz(mxmul,nnuc),Sx(mxmul,nnuc),SSxy(mxmul,nnuc),
     .                SzS(mxmul,nnuc),Q(3,nnuc)
*---> local
      DOUBLE PRECISION RI,RI2,KK,eta
*
* INIT
      call dzero(Sz,mxmul*nnuc)
      call dzero(Sx,mxmul*nnuc)
      call dzero(SSxy,mxmul*nnuc)
      call dzero(SzS,mxmul*nnuc)
*
      DO 400 i = 1 , nnuc
      if(mult(i).lt.3) goto 400
*
      RI = DBLE(mult(i)-1)/2.D0
      RI2= RI*(RI+1.D0)
      if(DABS(Q(3,i)).lt.1.D-8) then
         eta = 0.D0
      else
         eta= DABS( (Q(1,i)-Q(2,i))/Q(3,i) )
      endif
      KK = Q(3,i)/2.D0
*
      DO 100 k = 1 , mult(i)
      Sz(k,i) = DBLE(mult(i)-2*k+1)/2.D0
  100 SzS(k,i) = ( 3*Sz(k,i)**2-RI2 ) * KK
      DO 200 k = 1 , mult(i)-1
  200 Sx(k,i) = DSQRT(DBLE(k*(mult(i)-k)))/2.D0
      DO 300 k = 1 , mult(i)-2
  300 SSxy(k,i) = Sx(k,i)*Sx(k+1,i)*2.D0*KK*eta
*
  400 continue
      END

*----------------------------------------------------------------------
      SUBROUTINE spinZE(Sx,Sz,SSxy,SzS,mult,HX,HY,HZ,R,
     .                  DHR,DHI,W,E,VR,VI)
*----------------------------------------------------------------------
*     CALCULATION OF HAM-MATRIX FOR SPIN IN ZEEMAN FIELD AND ZFI/NQI
*>>>>> TO BE USED IN CONJUCTION WITH ROUTINE spinZF(...) WHICH WILL
*>>>>> SET UP THE NECESSARY SPIN MATRIX COMPONENTS (SEE COMMENTS THERE)
*
*     mult    : multiplicity of spin
*     DX/DY/DZ: vector of effective zeeman interaction (10-4 cm-1)
*     R       : Principle directions of Q (direction cosines)
*
*     WORKSPACE PASSED FROM CALLING ROUTINE:
*     DHR/DHI(mult,mult): work space H-matrix (REAL/IMAG-part)
*     W(3,mult)         : work arrays for diagonalization routine
*
*     OUTPUT
*     E(mult): vector of eigenvalues
*     VR(mult,mult): eigenvector matrix (real part)
*     VI(mult,mult): eigenvector matrix (imag part)
*
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ
      PARAMETER(XX=1,XY=2,XZ=3,YX=4,YY=5,YZ=6,ZX=7,ZY=8,ZZ=9)
      INTEGER mult,k,ifail,i
      DOUBLE PRECISION Sz(mult),Sx(mult),SSxy(mult),SzS(mult),Hx,Hy,Hz,
     .                 R(9),DHR(mult,mult),DHI(mult,mult),W(mult,3),
     .                 E(mult),VR(mult,mult),VI(mult,mult),Dx,Dy,Dz
*
      if(mult.lt.3) return
      call dzero(dhr,mult*mult)
      call dzero(dhi,mult*mult)
      call dzero(vr ,mult*mult)
      call dzero(vi ,mult*mult)
      call dzero(w  ,mult*3   )
*
*     CALCULATION OF DX,DY,DZ IN PRINCIPLE AXIS SYSTEM OF Q
*
      DX =  HX * R(XX) + HY * R(YX) + HZ * R(ZX)
      DY =  HX * R(XY) + HY * R(YY) + HZ * R(ZY)
      DZ =  HX * R(XZ) + HY * R(YZ) + HZ * R(ZZ)
*
*---> sub1-diagonal : zeeman contributions
      DO 100 k = 1 , mult-1
         DHR(k,k+1) = Dx*Sx(k)
         DHR(k+1,k) = Dx*Sx(k)
         DHI(k,k+1) = -Dy*Sx(k)
         DHI(k+1,k) =  Dy*Sx(k)
  100 CONTINUE
*---> diagonal: zeeman + K contributions
      DO 200 k = 1 , mult
  200    DHR(k,k) = Dz*Sz(k) + SzS(k)
*---> sub2-diagonal : K*eta contributions
      DO 300 k = 1 , mult-2
         DHR(k,k+2) = SSxy(k)
         DHR(k+2,k) = SSxy(k)
  300 CONTINUE
*---> diagonalize H-matrix
*check::::
*     write(6,*) ' H-matrix into HERMEV:'
*     write(6,'(6(1X,G11.4))') ((DHR(i,k),DHI(i,k),i=1,3),k=1,3)

      CALL HERMEV(mult,mult,DHR,DHI,1,VR,VI,E,W,W(1,2),ifail)
*
      if(ifail.ne.0) stop ' HERMEV diagonalization failed !!!'
      END
*
*----------------------------------------------------------------------
*     DIAGONALIZATION ROUTINE (EISPACK)
*----------------------------------------------------------------------
      SUBROUTINE HERMEV(NM,N,AR,AI,MATZ,ZR,ZI,D,E,TAU,IERR)
*----------------------------------------------------------------------
      REAL*8 AR(NM,N), AI(NM,N), ZR(NM,N),ZI(NM,N)
      REAL*8 E(N), TAU(2,N), D(N)
      INTEGER MATZ

      CALL HTRIDI(NM,N,AR,AI,D,E,E,TAU)
      DO 200 I=1,N
      DO 200 J=1,N
      ZR(I,J)=0.0D0
      IF(I.EQ.J) ZR(I,J)=1.0D0
  200 CONTINUE
      CALL TQL2(NM,N,D,E,ZR,IERR)
      IF(MATZ.EQ.0) RETURN
      CALL HTRIBK(NM,N,AR,AI,TAU,N,ZR,ZI)
      END

C***********************************************************************
      SUBROUTINE  HERMIT(NM,N,A,Z,AR,AI,ZR,ZI,D,E,TAU,IERR)
C
C     ROUTINE FOR THE DIAGONALIZATION OF A COMPLEX HERMITIAN MATRIX
C
C
C     A  CONTAINS THE INPUTMATRIX
C     D AND B  CONTAIN THE EIGENVALUES AND EIGENVECTORS
C
C     THE ARRAY A MAY BE EQUIVALENCED WITH THE ARRAYS ZR AND ZI
C     THE ARRAY Z MAY BE EQUIVALENCED WITH THE ARRAYS AR AND AI
C
C         FOR EXAMPLE CALL AS FOLLOWS :
C     CALL HERMIT(30,18,A,Z,Z(1,1),Z(1,16),A(1,1),A(1,16),D,E,TAU,IERR)
C
C
C     FOR A FURTHER DESCRIPTION SEE THE SUBROUTINES
C         HTRIDI , HTRIBK AND TQL2
C
C
C     ------------------------------------------------------------------
C
      IMPLICIT REAL*8  (A-H,O-Z)
      COMPLEX*16  A(NM,NM),Z(NM,NM)
      DIMENSION  AR(NM,NM),AI(NM,NM),D(N),E(N),TAU(2,N),ZR(NM,NM),
     *           ZI(NM,NM)
      DO 100 I=1,N
      DO 100 J=1,N
      AR(I,J)=DREAL(A(I,J))
  100 AI(I,J)=DIMAG(A(I,J))
      CALL HTRIDI(NM,N,AR,AI,D,E,E,TAU)
      DO 200 I=1,N
      DO 200 J=1,N
      ZR(I,J)=0.0D0
      IF(I.EQ.J) ZR(I,J)=1.0D0
  200 CONTINUE
      CALL TQL2(NM,N,D,E,ZR,IERR)
      CALL HTRIBK(NM,N,AR,AI,TAU,N,ZR,ZI)
      DO 300 I=1,N
      DO 300 J=1,N
  300 Z(I,J)=DCMPLX(ZR(I,J),ZI(I,J))
      RETURN
      END
      SUBROUTINE HTRIDI(NM,N,AR,AI,D,E,E2,TAU)
C
C     ------------------------------------------------------------------
C
      IMPLICIT REAL*8  (A-H,O-Z)
      DIMENSION  AR(NM,N),AI(NM,N),D(N),E(N),E2(N),TAU(2,N)
C
C
C     FOR THE SOLUTION OF A COMPLEX HERMITIAN EIGENVALUE PROBLEM
C
C     SUBROUTINES    HTRIDI  HTRIBK  AND TQL2
C
C     ARE NEEDED
C
C      DIMENSION AR(NM,N),AI(NM,N),D(N),E(N),TAU(2,N,ZR(NM,N),ZI(NM,N)
C
C     PLACE THE REAL AND IMAGINARY PARTS OF THE MATRIX IN
C      AR  AND AI
C     FILL THE MATRIX ZR WITH THE IDENTITY
C     DECLARATIONS NEEDED
C
C      THEN
C     CALL HTRIDI(NM,N,AR,AI,D,E,E,TAU)
C     CALL TQL2(NM,N,D,E,ZR,IERR)
C     CALL HTRIBK(NM,N,AR,AI,TAU,N,ZR,ZI)
C
C
C     D CONTAINS THE EIGENVALUES
C     ZR AND ZI CONTAINS RESP. THE REAL AND IMAGINARY PARTS OF
C     THE COLUMNS OF THE EIGENVECTORS
C
C     IERR IS THE ERROR CODE  AND MUST BE ZERO
C     IF IERR IS NOT ZERO , SEE COMMENT OF TQL2
C
C
C     REAL*8  DSQRT,CDABS,DABS
C     COMPLEX*16  DCMPLX
C
C     THIS SUBROUTINE IS A TRANSLATION OF A COMPLEX ANALOGUE OF
C     THE ALGOL PROCEDURE TRED1, NUM. MATH. 11, 181-195(1968)
C     BY MARTIN, REINSCH, AND WILKINSON.
C     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971).
C
C     THIS SUBROUTINE REDUCES A COMPLEX HERMITIAN MATRIX
C     TO A REAL SYMMETRIC TRIDIAGONAL MATRIX USING
C     UNITARY SIMILARITY TRANSFORMATIONS.
C
C     ON INPUT-
C
C        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL
C          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM
C          DIMENSION STATEMENT,
C
C        N IS THE ORDER OF THE MATRIX,
C
C        AR AND AI CONTAIN THE REAL AND IMAGINARY PARTS,
C          RESPECTIVELY, OF THE COMPLEX HERMITIAN INPUT MATRIX.
C          ONLY THE LOWER TRIANGLE OF THE MATRIX NEED BE SUPPLIED.
C
C     ON OUTPUT-
C
C        AR AND AI CONTAIN INFORMATION ABOUT THE UNITARY TRANS-
C          FORMATIONS USED IN THE REDUCTION IN THEIR FULL LOWER
C          TRIANGLES.  THEIR STRICT UPPER TRIANGLES AND THE
C          DIAGONAL OF AR ARE UNALTERED,
C
C        D CONTAINS THE DIAGONAL ELEMENTS OF THE THE TRIDIAGONAL MATRIX,
C
C        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE TRIDIAGONAL
C          MATRIX IN ITS LAST N-1 POSITIONS.  E(1) IS SET TO ZERO,
C
C        E2 CONTAINS THE SQUARES OF THE CORRESPONDING ELEMENTS OF E.
C          E2 MAY COINCIDE WITH E IF THE SQUARES ARE NOT NEEDED,
C
C        TAU CONTAINS FURTHER INFORMATION ABOUT THE TRANSFORMATIONS.
C
C     ARITHMETIC IS REAL EXCEPT FOR THE USE OF THE SUBROUTINES
C     CDABS AND DCMPLX IN COMPUTING COMPLEX ABSOLUTE VALUES
C
C
C     ------------------------------------------------------------------
C
      TAU(1,N) = 1.0
      TAU(2,N) = 0.0
C
      DO 100 I = 1, N
  100 D(I) = AR(I,I)
C
C     ********** FOR I=N STEP -1 UNTIL 1 DO -- **********
      DO  300 II = 1, N
         I = N + 1 - II
         L = I - 1
         H = 0.0
         SCALE = 0.0
         IF (L .LT. 1) GO TO 130
C     ********** SCALE ROW (ALGOL TOL THEN NOT NEEDED) **********
         DO 120 K = 1, L
  120    SCALE = SCALE +DABS(AR(I,K)) +DABS(AI(I,K))
C
         IF (SCALE .NE. 0.0) GO TO 140
         TAU(1,L) = 1.0
         TAU(2,L) = 0.0
  130    E(I) = 0.0
         E2(I) = 0.0
         GO TO 290
C
  140    DO 150 K = 1, L
            AR(I,K) = AR(I,K) / SCALE
            AI(I,K) = AI(I,K) / SCALE
            H = H + AR(I,K) * AR(I,K) + AI(I,K) * AI(I,K)
  150    CONTINUE
C
         E2(I) = SCALE * SCALE * H
         G =DSQRT(H)
         E(I) = SCALE * G
         F =CDABS(DCMPLX(AR(I,L),AI(I,L)))
C     ********** FORM NEXT DIAGONAL ELEMENT OF MATRIX T **********
         IF (F .EQ. 0.0) GO TO 160
         TAU(1,L) = (AI(I,L) * TAU(2,I) - AR(I,L) * TAU(1,I)) / F
         SI = (AR(I,L) * TAU(2,I) + AI(I,L) * TAU(1,I)) / F
         H = H + F * G
         G = 1.0 + G / F
         AR(I,L) = G * AR(I,L)
         AI(I,L) = G * AI(I,L)
         IF (L .EQ. 1) GO TO 270
         GO TO 170
  160    TAU(1,L) = -TAU(1,I)
         SI = TAU(2,I)
         AR(I,L) = G
  170    F = 0.0
C
         DO 240 J = 1, L
            G = 0.0
            GI = 0.0
C     ********** FORM ELEMENT OF A*U **********
            DO 180 K = 1, J
               G = G + AR(J,K) * AR(I,K) + AI(J,K) * AI(I,K)
               GI = GI - AR(J,K) * AI(I,K) + AI(J,K) * AR(I,K)
  180       CONTINUE
C
            JP1 = J + 1
            IF (L .LT. JP1) GO TO 220
C
            DO 200 K = JP1, L
               G = G + AR(K,J) * AR(I,K) - AI(K,J) * AI(I,K)
               GI = GI - AR(K,J) * AI(I,K) - AI(K,J) * AR(I,K)
  200       CONTINUE
C     ********** FORM ELEMENT OF P **********
  220       E(J) = G / H
            TAU(2,J) = GI / H
            F = F + E(J) * AR(I,J) - TAU(2,J) * AI(I,J)
  240    CONTINUE
C
         HH = F / (H + H)
C     ********** FORM REDUCED A **********
         DO 260 J = 1, L
            F = AR(I,J)
            G = E(J) - HH * F
            E(J) = G
            FI = -AI(I,J)
            GI = TAU(2,J) - HH * FI
            TAU(2,J) = -GI
C
            DO 260 K = 1, J
               AR(J,K) = AR(J,K) - F * E(K) - G * AR(I,K)
     X                           + FI * TAU(2,K) + GI * AI(I,K)
               AI(J,K) = AI(J,K) - F * TAU(2,K) - G * AI(I,K)
     X                           - FI * E(K) - GI * AR(I,K)
  260    CONTINUE
C
  270    DO 280 K = 1, L
            AR(I,K) = SCALE * AR(I,K)
            AI(I,K) = SCALE * AI(I,K)
  280    CONTINUE
C
         TAU(2,L) = -SI
  290    HH = D(I)
         D(I) = AR(I,I)
         AR(I,I) = HH
         AI(I,I) = SCALE * SCALE * H
  300 CONTINUE
C
      RETURN
      END
      SUBROUTINE HTRIBK(NM,N,AR,AI,TAU,M,ZR,ZI)
C
C     ------------------------------------------------------------------
C
      IMPLICIT REAL*8  (A-H,O-Z)
      DIMENSION  AR(NM,N),AI(NM,N),TAU(2,N),ZR(NM,M),ZI(NM,M)
C
C     THIS SUBROUTINE IS A TRANSLATION OF A COMPLEX ANALOGUE OF
C     THE ALGOL PROCEDURE TRBAK1, NUM. MATH. 11, 181-195(1968)
C     BY MARTIN, REINSCH, AND WILKINSON.
C     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971).
C
C     THIS SUBROUTINE FORMS THE EIGENVECTORS OF A COMPLEX HERMITIAN
C     MATRIX BY BACK TRANSFORMING THOSE OF THE CORRESPONDING
C     REAL SYMMETRIC TRIDIAGONAL MATRIX DETERMINED BY  HTRIDI.
C
C     ON INPUT-
C
C        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL
C          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM
C          DIMENSION STATEMENT,
C
C        N IS THE ORDER OF THE MATRIX,
C
C        AR AND AI CONTAIN INFORMATION ABOUT THE UNITARY TRANS-
C          FORMATIONS USED IN THE REDUCTION BY  HTRIDI  IN THEIR
C          FULL LOWER TRIANGLES EXCEPT FOR THE DIAGONAL OF AR.
C
C        TAU CONTAINS FURTHER INFORMATION ABOUT THE TRANSFORMATIONS,
C
C        M IS THE NUMBER OF EIGENVECTORS TO BE BACK TRANSFORMED,
C
C        ZR CONTAINS THE EIGENVECTORS TO BE BACK TRANSFORMED
C          IN ITS FIRST M COLUMNS.
C
C     ON OUTPUT-
C
C        ZR AND ZI CONTAIN THE REAL AND IMAGINARY PARTS,
C          RESPECTIVELY, OF THE TRANSFORMED EIGENVECTORS
C          IN THEIR FIRST M COLUMNS.
C
C     NOTE THAT THE LAST COMPONENT OF EACH RETURNED VECTOR
C     IS REAL AND THAT VECTOR EUCLIDEAN NORMS ARE PRESERVED.
C
C
C     ------------------------------------------------------------------
C
C     ********** TRANSFORM THE EIGENVECTORS OF THE REAL SYMMETRIC
C                TRIDIAGONAL MATRIX TO THOSE OF THE HERMITIAN
C                TRIDIAGONAL MATRIX. **********
      DO 50 K = 1, N
C
         DO 50 J = 1, M
            ZI(K,J) = - ZR(K,J) * TAU(2,K)
            ZR(K,J) = ZR(K,J) * TAU(1,K)
   50 CONTINUE
C
      IF (N .EQ. 1) GO TO 200
C     ********** RECOVER AND APPLY THE HOUSEHOLDER MATRICES **********
      DO 140 I = 2, N
         L = I - 1
         H = AI(I,I)
         IF (H .EQ. 0.0) GO TO 140
C
         DO 130 J = 1, M
            S = 0.0
            SI = 0.0
C
            DO 110 K = 1, L
               S = S + AR(I,K) * ZR(K,J) - AI(I,K) * ZI(K,J)
               SI = SI + AR(I,K) * ZI(K,J) + AI(I,K) * ZR(K,J)
  110       CONTINUE
C
            S = S / H
            SI = SI / H
C
            DO 120 K = 1, L
               ZR(K,J) = ZR(K,J) - S * AR(I,K) - SI * AI(I,K)
               ZI(K,J) = ZI(K,J) - SI * AR(I,K) + S * AI(I,K)
  120       CONTINUE
C
  130    CONTINUE
C
  140 CONTINUE
C
  200 RETURN
      END
      SUBROUTINE TQL2(NM,N,D,E,Z,IERR)
C
C     ------------------------------------------------------------------
C
      IMPLICIT REAL*8  (A-H,O-Z)
      DIMENSION  D(N),E(N),Z(NM,N)
      REAL*8  MACHEP
C
C     REAL*8  DSQRT,DABS,DSIGN
C
C     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQL2,
C     NUM. MATH. 11, 293-306(1968) BY BOWDLER, MARTIN, REINSCH, AND
C     WILKINSON.
C     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 227-240(1971).
C
C     THIS SUBROUTINE FINDS THE EIGENVALUES AND EIGENVECTORS
C     OF A SYMMETRIC TRIDIAGONAL MATRIX BY THE QL METHOD.
C     THE EIGENVECTORS OF A FULL SYMMETRIC MATRIX CAN ALSO
C     BE FOUND IF  TRED2  HAS BEEN USED TO REDUCE THIS
C     FULL MATRIX TO TRIDIAGONAL FORM.
C
C     ON INPUT-
C
C        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL
C          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM
C          DIMENSION STATEMENT,
C
C        N IS THE ORDER OF THE MATRIX,
C
C        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX,
C
C        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX
C          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY,
C
C        Z CONTAINS THE TRANSFORMATION MATRIX PRODUCED IN THE
C          REDUCTION BY  TRED2, IF PERFORMED.  IF THE EIGENVECTORS
C          OF THE TRIDIAGONAL MATRIX ARE DESIRED, Z MUST CONTAIN
C          THE IDENTITY MATRIX.
C
C      ON OUTPUT-
C
C        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN
C          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT BUT
C          UNORDERED FOR INDICES 1,2,...,IERR-1,
C
C        E HAS BEEN DESTROYED,
C
C        Z CONTAINS ORTHONORMAL EIGENVECTORS OF THE SYMMETRIC
C          TRIDIAGONAL (OR FULL) MATRIX.  IF AN ERROR EXIT IS MADE,
C          Z CONTAINS THE EIGENVECTORS ASSOCIATED WITH THE STORED
C          EIGENVALUES,
C
C        IERR IS SET TO
C          ZERO       FOR NORMAL RETURN,
C          J          IF THE J-TH EIGENVALUE HAS NOT BEEN
C                     DETERMINED AFTER 30 ITERATIONS.
C
C
C     ------------------------------------------------------------------
C
C     ********** MACHEP IS A MACHINE DEPENDENT PARAMETER SPECIFYING
C                THE RELATIVE PRECISION OF FLOATING POINT ARITHMETIC.
C
C                **********
      MACHEP = 2.**(-47)
C
      IERR = 0
      IF (N .EQ. 1) GO TO 1001
C
      DO 100 I = 2, N
  100 E(I-1) = E(I)
C
      F = 0.0
      B = 0.0
      E(N) = 0.0
C
      DO 240 L = 1, N
         J = 0
         H = MACHEP *(DABS(D(L)) +DABS(E(L)))
         IF (B .LT. H) B = H
C     ********** LOOK FOR SMALL SUB-DIAGONAL ELEMENT **********
         DO 110 M = L, N
            IF(DABS(E(M)) .LE. B) GO TO 120
C     ********** E(N) IS ALWAYS ZERO, SO THERE IS NO EXIT
C                THROUGH THE BOTTOM OF THE LOOP **********
  110    CONTINUE
C
  120    IF (M .EQ. L) GO TO 220
  130    IF (J .EQ. 30) GO TO 1000
         J = J + 1
C     ********** FORM SHIFT **********
         P = (D(L+1) - D(L)) / (2.0 * E(L))
         R =DSQRT(P*P+1.0)
         H = D(L) - E(L) / (P +DSIGN(R,P))
C
         DO 140 I = L, N
  140    D(I) = D(I) - H
C
         F = F + H
C     ********** QL TRANSFORMATION **********
         P = D(M)
         C = 1.0
         S = 0.0
         MML = M - L
C     ********** FOR I=M-1 STEP -1 UNTIL L DO -- **********
         DO 200 II = 1, MML
            I = M - II
            G = C * E(I)
            H = C * P
            IF(DABS(P) .LT.DABS(E(I))) GO TO 150
            C = E(I) / P
            R =DSQRT(C*C+1.0)
            E(I+1) = S * P * R
            S = C / R
            C = 1.0 / R
            GO TO 160
  150       C = P / E(I)
            R =DSQRT(C*C+1.0)
            E(I+1) = S * E(I) * R
            S = 1.0 / R
            C = C * S
  160       P = C * D(I) - S * G
            D(I+1) = H + S * (C * G + S * D(I))
C     ********** FORM VECTOR **********
            DO 180 K = 1, N
               H = Z(K,I+1)
               Z(K,I+1) = S * Z(K,I) + C * H
               Z(K,I) = C * Z(K,I) - S * H
  180       CONTINUE
C
  200    CONTINUE
C
         E(L) = S * P
         D(L) = C * P
         IF(DABS(E(L)) .GT. B) GO TO 130
  220    D(L) = D(L) + F
  240 CONTINUE
C     ********** ORDER EIGENVALUES AND EIGENVECTORS **********
      DO 300 II = 2, N
         I = II - 1
         K = I
         P = D(I)
C
         DO 260 J = II, N
            IF (D(J) .GE. P) GO TO 260
            K = J
            P = D(J)
  260    CONTINUE
C
         IF (K .EQ. I) GO TO 300
         D(K) = D(I)
         D(I) = P
C
         DO 280 J = 1, N
            P = Z(J,I)
            Z(J,I) = Z(J,K)
            Z(J,K) = P
  280    CONTINUE
C
  300 CONTINUE
C
      GO TO 1001
C     ********** SET ERROR -- NO CONVERGENCE TO AN
C                EIGENVALUE AFTER 30 ITERATIONS **********
 1000 IERR = L
 1001 RETURN
      END
      FUNCTION DREAL(Z)
C---------DREAL AND DIMAG GIVE THE REAL AND THE IMAGINARY PART
C---------OF A COMPLEX*16 NUMBER.
      REAL*8 DREAL,DIMAG,X(2)
      COMPLEX*16 Z,W
      EQUIVALENCE  (W,X(1))
      W=Z
      DREAL=X(1)
      RETURN
      ENTRY    DIMAG(Z)
      W=Z
      DIMAG=X(2)
      RETURN
      END

*======================================================================
*     Utilities: manipulating with vector matrices and tensors
*= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
*
*------------------------------------------------------------------
      SUBROUTINE VNORM(VR,VI,N)
*------------------------------------------------------------------
*     NORMALIZE EIGENVECTOR MATRIX
      IMPLICIT NONE
      INTEGER*4 I,J,N
      DOUBLE PRECISION VR(N,N),VI(N,N),norm
*
      DO 50 J = 1,N
      NORM = 0.0D0
      DO 20 I = 1,N
   20 NORM = NORM + VR(I,J)*VR(I,J) + VI(I,J)*VI(I,J)
      norm = dsqrt(norm)
*     write(6,'('' norm:'',i4,1x,g10.4)') j, norm
      DO 25 I = 1,N
      VR(I,J) = VR(I,J) / NORM
   25 VI(I,J) = VI(I,J) / NORM
   50 CONTINUE
*
      END

*--------------------------------------------------------------
      SUBROUTINE INPROD(VR1,VI1,VR2,VI2,RPROD,IPROD,N)
*--------------------------------------------------------------
*     INPRODUCT MATRIX OF TWO EIGENVECTOR MATRICES V1(+)*V2
      IMPLICIT NONE
      INTEGER*4 I,J,K,N
      DOUBLE PRECISION VR1(N,N),VR2(N,N),VI1(N,N),VI2(N,N),
     -                 RPROD(N,N),IPROD(N,N),RPRO,IPRO
*
      DO 100 I = 1,N
      DO 100 J = 1,N
          RPRO = 0.0D0
          IPRO = 0.0D0
          DO 50 K = 1,N
            RPRO = RPRO + VR1(K,I)*VR2(K,J) + VI1(K,I)*VI2(K,J)
            IPRO = IPRO + VR1(K,I)*VI2(K,J) - VI1(K,I)*VR2(K,J)
   50     CONTINUE
          RPROD(I,J) = RPRO
          IPROD(I,J) = IPRO
  100 CONTINUE
*
      END
*-----------------------------------------------------------------
      SUBROUTINE COMPLX(VR,VI,VMAG,VANG,N,IMODE)
*-----------------------------------------------------------------
*     CONVERSION OF COMPLEX NUMBERS INTO MAGNITUDE**2 AND POLAR
*     ANGLE AND VICE VERZA
*
      IMPLICIT NONE
      INTEGER*4 I,N,IMODE
      DOUBLE PRECISION VR(N),VI(N),VMAG(N),VANG(N),MAG,ANG,REAL,IMAG
*
      IF(IMODE.EQ.-1) GOTO 2
      DO 100 I = 1,N
      MAG = VR(I)*VR(I) + VI(I)*VI(I)
      ANG = DATAN2(VI(I),VR(I))
      VMAG(I) = MAG
      VANG(I) = ANG
  100 CONTINUE
      RETURN
*
*--------- REVERSE
*
    2 CONTINUE
      DO 200 I = 1,N
      REAL = DSQRT(VMAG(I))*DCOS(VANG(I))
      IMAG = DSQRT(VMAG(I))*DSIN(VANG(I))
      VR(I) = REAL
      VI(I) = IMAG
  200 CONTINUE
      RETURN
      END

*--------------------------------------------------------------------
      subroutine euler2cos(euler,Rcos)
*--------------------------------------------------------------------
      IMPLICIT none
      REAL*8 euler(3),Rcos(3,3),pi2pi,al,be,ga
      pi2pi = dacos(-1.D0)*2d0/360d0
*
      al = euler(1)*pi2pi
      be = euler(2)*pi2pi
      ga = euler(3)*pi2pi
      Rcos(1,1) =  DCOS(AL)*DCOS(BE)*DCOS(GA) - DSIN(AL)*DSIN(GA)
      Rcos(1,2) =  DSIN(AL)*DCOS(BE)*DCOS(GA) + DCOS(AL)*DSIN(GA)
      Rcos(1,3) = -DSIN(BE)*DCOS(GA)
      Rcos(2,1) = -DCOS(AL)*DCOS(BE)*DSIN(GA) - DSIN(AL)*DCOS(GA)
      Rcos(2,2) = -DSIN(AL)*DCOS(BE)*DSIN(GA) + DCOS(AL)*DCOS(GA)
      Rcos(2,3) =  DSIN(BE)*DSIN(GA)
      Rcos(3,1) =  DCOS(AL)*DSIN(BE)
      Rcos(3,2) =  DSIN(AL)*DSIN(BE)
      Rcos(3,3) =  DCOS(BE)
      end

*-----------------------------------------------------------------------
      subroutine Trot(PV,Rmat,tensor)
*-----------------------------------------------------------------------
*
*     SUBROUTINE TO MULTIPLY A TENSOR WITH A MATRIX T'=R.T.R(-1)
*
      IMPLICIT none
      REAL*8 PV(3), Rmat(3,3), tensor(6), e(3)
      INTEGER*4 i,j,k,n
*...  save principle values (enable equivalence of pv, and tensor)
      do 10 i = 1,3
 10   e(i) = pv(i)
      k = 0
      DO 100 j=1,3
      DO 100 n=1,j
      k = k + 1
      tensor(k)    = Rmat(n,1)*Rmat(j,1)*e(1) +
     +               Rmat(n,2)*Rmat(j,2)*e(2) +
     +               Rmat(n,3)*Rmat(j,3)*e(3)
  100 CONTINUE
      END
*======================================================================
*     THE APPLE PEAL PROCEDURE:
*= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
*
**********************************
*     PROGRAM pealtest
*     DOUBLE PRECISION theta,phi
*     INTEGER krid,ncall
*     LOGICAL last
*     ncall = 0
*     krid = 10
*  1  call pwdpeal(krid,theta,phi,ncall,last)
*     write(6,*) ncall, theta, phi, last
*     if(.not.last) goto 1
*     end
*----------------------------------------------------------------------
      SUBROUTINE pwdpeal(krid,thetap,phip,ncall,last)
*----------------------------------------------------------------------
* appel peal algorithm to integrate of sphere
*
      IMPLICIT none
      DOUBLE PRECISION thetap,phip,step,thetaa,dthe,dphi,thetam,pi
      INTEGER krid,ncall,itheta,iphi,nphi
      LOGICAL last
      SAVE itheta,iphi,nphi,step,thetaa,dthe,dphi,thetam
      DATA pi /3.141592653589793D0/
      DATA step/0.d0/
      if(step.eq.0.d0.and.ncall.ne.0) stop ' PWDPEAL unitialized!'
      if(krid.le.0) stop ' PWDPEAL illegal value of krid'
      if(ncall.ne.0) goto 111
           itheta = krid
           iphi   = 0
           step = 90D0 / dble(krid)
           nphi = 2*krid
           thetaa = 90d0
           dthe = -0.5d0 * step / dble(nphi)
           dphi = 360D0 / dble(nphi)
  111 continue
      if(itheta.le.0) stop ' pdrpeal too many calls'
      if(iphi.eq.nphi) then
           iphi = 0
           itheta = itheta -1
           thetam = dble(itheta) * step
           nphi = idnint(dsin(pi/180d0*thetam)*dble(4*krid))
           if(nphi.le.0) STOP 'BUG pdrpeal 1'
*              SMALLEST NPHI IS TYPICALLY 6
           thetaa = thetam + 0.5d0 * step
           dthe = - step / dble(nphi)
           dphi = 360D0 / dble(nphi)
      endif
      thetap = thetaa + dble(iphi) * dthe
      phip   = dble(iphi) * dphi
      iphi  = iphi + 1
      ncall = ncall + 1
      last = (iphi.eq.nphi.and.itheta.eq.1)
      if(last) step = 0.d0
      end
