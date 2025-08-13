*     DOUBLE PRECISION offset,factor
*     DIMENSION exp(200), xmod(200)
*     DO 10 I = 1 , 200
*     xmod(i) = cos(i*3.14/33.0)
* 10  exp(i) = xmod(i)*1.23 + 3.21
*     call xscale(exp,xmod,100,offset,factor)
*     write(6,*) offset,factor
*     end
*
*     logical ok
*     integer ival(10),vindex(10),i1,i2
*     double precision rval(10)
*     data i1,i2 /0,0/
*   1 call userin(nwords)
*     ok = .true.
*     call getrange(1,i1,i2,ok)
*     write(6,*) i1,i2
*     if(i1.ne.0.or.i2.ne.0) goto 1
*     end
**INCLUDE USRIN.FOR
*
*----------------------------------------------------------------------
      SUBROUTINE xscale(exp,xmod,n,offset,factor)
*----------------------------------------------------------------------
*     calculate offset and scale for model spectrum fit to experiment
*
*     exp:  experimental array        n:       array length
*     xmod: model array               offset:  offset
*                                     factor:  scaling factor
*
*     Algorithm based on linear regression:
*
*     /             \ /    \     /    \
*     |  n    SMOD  | | a0 |  =  | SX |
*     | SMOD  SMOD2 | | a1 |     | SXY|
*     \             / \    /     \    /
*
*        SMOD = SUM (xmod)       SX = SUM ( exp )
*        SMOD2= SUM (xmod**2)    SXY= SUM ( exp(i)*xmod(i) )
*
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER i,n
      REAL exp(n),xmod(n)
      DOUBLE PRECISION offset,factor,smod,smod2,sx,sxy
*
      smod  = 0.D0
      smod2 = 0.D0
      sx    = 0.D0
      sxy   = 0.D0
      DO 100  i = 1 , n
         smod  = smod + xmod(i)
         smod2 = smod2 + xmod(i)**2
         sx    = sx + exp(i)
         sxy   = sxy + exp(i)*xmod(i)
 100  CONTINUE
      offset = ( sx*smod2 - sxy*smod )/( smod2*n - smod**2 )
      factor = ( sx*smod  - sxy*n    )/( smod**2 - smod2*n )
      END

*======================================================================
*     ARRAY manipulation
*
      SUBROUTINE vcopy(ara,arb,const,nar)
      DOUBLE PRECISION ara(*),arb(*),const
      DO 10 i = 1 , nar
  10  arb(I) = ara(I) * const
      END

      SUBROUTINE rcopy(raya,rayb,nar)
      DOUBLE PRECISION raya(*),rayb(*)
      DO 10 i = 1 , nar
  10  rayb(i) = raya(i)
      END

      SUBROUTINE rminmax(ray,nar,ymin,ymax)
      DOUBLE PRECISION ymin, ymax
      REAL ray(*)
      if(nar.le.0) return
      amin = ray(1)
      amax = ray(1)
      DO 10 i = 1 , nar
      amax = amax1(amax,ray(i))
  10  amin = amin1(amin,ray(i))
      ymin = amin
      ymax = amax
      END

      SUBROUTINE cminmax(cray,nar,ymin,ymax)
      DOUBLE PRECISION ymin, ymax
      COMPLEX cray(*)
      if(nar.le.0) return
      ymin = abs(cray(1))
      ymax = abs(cray(1))
      DO 10 i = 1 , nar
      amax = amax1(amax,abs(cray(i)))
  10  amin = amin1(amin,abs(cray(i)))
      ymin = amin
      ymax = amax
      END

      SUBROUTINE dminmax(ray,nar,ymin,ymax)
      DOUBLE PRECISION ymin, ymax, ray(*)
      if(nar.le.0) return
      ymin = ray(1)
      ymax = ray(1)
      DO 10 i = 1 , nar
      ymax = dmax1(ymax,ray(i))
  10  ymin = dmin1(ymin,ray(i))
      END

      FUNCTION dsum(dray,n)
      DOUBLE PRECISION dray(n),dsum
      dsum = 0.D0
      do 1 i = 1 , n
    1 dsum = dsum + dray(i)
      END

      FUNCTION rsum(rray,n)
      REAL rray(n),rsum
      rsum = 0.0
      do 1 i = 1 , n
    1 rsum = rsum + rray(i)
      END

      FUNCTION csum(cray,n)
      COMPLEX cray(n),csum
      csum = (0.0,0.0)
      do 1 i = 1 , n
    1 csum = csum + cray(i)
      END

      SUBROUTINE izero(iray,n)
      DIMENSION iray(n)
      do 1 i = 1 , n
    1 iray(i) = 0
      END

      SUBROUTINE rzero(array,nar)
      DIMENSION array(*)
      do 10 i = 1 , nar
  10  array(i) = 0.e0
      END

      SUBROUTINE dzero(dray,n)
      DOUBLE PRECISION dray(n)
      do 1 i = 1 , n
    1 dray(i) = 0.D0
      END

      SUBROUTINE lzero(lray,n)
      LOGICAL lray(n)
      do 1 i = 1 , n
    1 lray(i) = .false.
      END

      SUBROUTINE r8r4(dbuf,rbuf,n)
      DOUBLE PRECISION dbuf(*)
      DIMENSION rbuf(*)
      do 1 i = 1 , n
    1 rbuf(i) = dbuf(i)
      END
*
