************************************************************************
*---> POSTPROC-MODULE
*----------------------------------------------------------------------
      SUBROUTINE pproc(xmina,xmaxa,ifmta,rbufa,n1a,
     .                 xminb,xmaxb,ifmtb,rbufb,n1b)
*----------------------------------------------------------------------
*     INTERFACE between datapage driver and ROUTINE postproc
      IMPLICIT none
      REAL    xmina,xmaxa,xminb,xmaxb,rbufa(*),rbufb(*)
      INTEGER ifmta,ifmtb,n1a,n1b,junit
      DOUBLE PRECISION damp0,xdamp0,damp1,xdamp1
      INTEGER nskip,xnskip,ppmode,xppmode
      DATA nskip/0/,ppmode/0/,damp0/-1/,damp1/10.D0/

      CALL posproc(damp0,damp1,nskip,ppmode,
     .                   rbufa,n1a,xmina,xmaxa,ifmta,
     .                   rbufb,n1b,xminb,xmaxb,ifmtb)
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY setpp(xdamp0,xdamp1,xnskip,xppmode)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      damp0 = xdamp0
      damp1 = xdamp1
      nskip = xnskip
      ppmode = xppmode
      if(nskip.lt.0) stop ' setpp: illegal nskip'
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY listpp(junit)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      !write(junit,'('' DAMP0='',G10.3,'' DAMP1='',G10.3)')
      !      damp0,damp1
      !write(junit,'('' NSKIP='',I10  ,'' PMODE='',I10)') nskip,ppmode
      !write(junit,*) 'ppmode=(0,1,2): (nop,ift+skip+damp,+fft)'
      END
************************************************************************

*-----------------------------------------------------------------------
      SUBROUTINE posproc(damp0,damp1,nskip,ppmode,
     .                   rbufa,n1a,xmina,xmaxa,ifmta,
     .                   rbufb,n1b,xminb,xmaxb,ifmtb)
*-----------------------------------------------------------------------
*     ppmode (1:ift,skip,damp) (2:+fft)
*     rbufa and rbufb (+ connected constants) may be shared
*
      IMPLICIT none
      INTEGER ppmode,nskip,max,i,n1a,n1b,ifmta,ifmtb
      PARAMETER (max=1024)
      REAL xminb,xmaxb,xmina,xmaxa
      DOUBLE PRECISION damp0,damp1
      REAL work(2*max+16),rbufa(*),rbufb(*)
*---  local
      DOUBLE PRECISION r0,dx

      IF(ppmode.le.0) return
      IF(ifmta.ne.4)  return
      IF(n1a.le.0)  return
*
      IF(ppmode.ge.1) THEN
* ---   prepare rbuf for ifft
        rbufb(1) = rbufa(1)
        DO 100 i = n1a , 2 , -1
          rbufb(2*i-1) = 0.D0
          rbufb(2*i-2) = rbufa(i) / 2.0
  100   CONTINUE
        n1b = 2 * n1a
        rbufb(n1b) = 0.0
*
* ---   save dc-component:
        r0 = rbufb(1)
        rbufb(1) = 0.0
* ---   ifft
        CALL rffti(n1b,work)
        CALL rfftb(n1b,rbufb,work)
        dx = 1.D0/(2.D0*xmaxa)
        xmaxb = dble(n1b)*dx
*
* --- decay function of modulation
* -->>>>>discard aliazed half of time domain !!:
        n1b = n1b/2
        xmaxb = xmaxb/2.0
        if(damp1.le.0.D0) goto 201
        DO 200 i = 1,n1b
        rbufb(i) = rbufb(i) * DEXP(-(i-1)*dx/damp1)
  200   CONTINUE
*
  201   CONTINUE
* --- decay function of total signal
        if(damp0.lt.0.D0) goto 251
        DO 240 i = 1 , n1b*2
  240   rbufb(i) = rbufb(i) + r0
        if(damp0.eq.0.D0) goto 251
        DO 250 i = 1,n1b
        rbufb(i) = rbufb(i) * DEXP(-(i-1)*dx/damp0)
  250   CONTINUE
  251   CONTINUE

        if(nskip.eq.0) goto 401
        DO 300 i = 1,n1b-nskip
  300     rbufb(i) = rbufb(i+nskip)
*       DO 400 i = n1b-nskip+1,n1b
* 400     rbufb(i)=0.0
        n1b = n1b-nskip
        xminb = nskip*dx
*      (xmaxr is invariant...)
  401   CONTINUE
        ifmtb = 4
      ENDIF
*
      IF(ppmode.eq.2) THEN
*       zerofill
        DO 500 i = n1b+1 , (n1b+nskip)*2
  500      rbufb(i) = 0.0
        n1b = (n1b+nskip)*2
*
        call rfftf(n1b,rbufb,work)
*
        rbufb(1) = SQRT(rbufb(1)**2+rbufb(n1b)**2)/n1b
        DO 700 i = 2,n1b/2
  700      rbufb(i) = SQRT(rbufb(2*i-2)**2 + rbufb(2*i-1)**2)*2.0/n1b
        xminb = 0.D0
        xmaxb = 1.D0/(2.D0*dx)
        n1b = n1b/2
      ENDIF
      END
