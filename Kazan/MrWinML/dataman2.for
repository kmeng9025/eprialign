*---------------------------------------------------------------------
      SUBROUTINE DATAMEN(nwords,ip,maclevel,ircode)
*---------------------------------------------------------------------
      IMPLICIT none
      INTEGER nkeys,ipos,ip,nwords,index,noccur,maclevel,ircode,mxval
      PARAMETER(nkeys=16,mxval=20)
      INTEGER ip1,ip2,ival(mxval),vindexR(mxval),vindexI(mxval),lw,narg,
     .        junit
      DOUBLE PRECISION rval(mxval)
      LOGICAL ok
      CHARACTER*10 menu(nkeys)
      CHARACTER*8 format(5)
      CHARACTER*60 help(nkeys)
      CHARACTER*64 fname
*---> global
      INTEGER merror
      COMMON /macro/  merror
      DATA MENU /'PAGE      ','COPY      ','CLEAR     ','FORMAT    ',
     .           'OPEN      ','CLOSE     ','SAVE      ','LOAD      ',
     .           'EDIT      ','NORM      ','RMAX      ','RSUM      ',
     .           'ADD       ','SHAPE     ','HELP      ','RETURN    '/
      DATA format /'NATIVE  ','COLUMNS ','GRAPHER ',
     .             'SURFER  ','BES3T   '/
      DATA help
     ./' Current page (info)                                        ',
     . ' syntax: sourcepage targetpage                              ',
     . ' syntax: frompage topage                                    ',
     . ' syntax: NATIVE(default)/COLUMNS/GRAPHER/SURFER/BES3T       ',
     . ' syntax: junit filename                                     ',
     . ' syntax: junit                                              ',
     . ' syntax: frompage topage junit                              ',
     . ' syntax: topage junit                                       ',
     . ' syntax: page imin imax value                               ',
     . ' syntax: type page1 page2  (type:1=inprod;2=rms)            ',
     . ' syntax: page imin imax                                     ',
     . ' syntax: page imin imax                                     ',
     . ' syntax: sourcepage targetpage factor                       ',
     . ' syntax: page(/=0) type[1=G;2=GD;3=L;4=LD] width(x-units)   ',
     . ' Produces this text                                         ',
     . ' (Exit)                                                     '/
*
      ipos = ip
      ircode = 0
    1 continue
      ok = .true.
      goto(2,3,3) maclevel
    2 continue
*----> interpretation/definition level:
        IF(ip.eq.0) THEN
           call message(ircode)
           write(6,*) ' DATAMEN>'
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
           write(6,*) ' MACRO EXECUTION ABORTED, error-code=', merror
           maclevel = 0
           goto 9999
      endif
    9 continue
*=======>>> COMMAND DISPATCHER <<<<<===========
      goto(10,20,30,40,50,60,70,80,90,100,105,110,120,130,140,150)index
      write(6,*) ' illegal code DATAMEN',index
      stop
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  PAGE : info on spectra
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   10 continue
      if(nwords-ipos.lt.2) then
         ip1 = -1
         ip2 = -1
      else
         call getiray(ipos+1,ival,vindexI,2,ok)
         if(.not.ok) goto 1313
         call mderefI(ival,vindexI,2)
         ip1 = ival(1)
         ip2 = ival(2)
      endif
      call datapage(ip1,ip2)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  COPY : ip1 to ip2
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   20 continue
      call macparse(nwords,ipos+1,maclevel,index,2,0,ival,rval,
     .              vindexI,vindexR,narg)
      IF(narg.le.0) goto 1313
      call mderefI(ival,vindexI,2)
      call datacopy(ival(1),ival(2))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  CLEAR: ip1 to ip2
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   30 continue
      call macparse(nwords,ipos+1,maclevel,index,2,0,ival,rval,
     .              vindexI,vindexR,narg)
      IF(narg.le.0) goto 1313
      call mderefI(ival,vindexI,2)
      call dataclear(ival(1),ival(2))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  FORMAT
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   40 continue
      CALL recog(ipos+1,format,5,index,noccur)
      if(noccur.ne.1) goto 1313
      call datform(index)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  OPEN junit filename
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   50 continue
      call getint(ipos+1,junit,ok)
      if(.not.ok) goto 1313
      call gtfname(ipos+2,nwords,fname,lw)
      if(lw.le.0) goto 1313
      OPEN(junit,FILE=fname(1:lw),err=1314)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  CLOSE junit
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   60 continue
      call getint(ipos+1,junit,ok)
      if(.not.ok) goto 1313
      CLOSE (junit)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> SAVE  ip1,ip2,junit
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   70 continue
      call macparse(nwords,ipos+1,maclevel,index,3,0,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,3)
      call datasave(ival(1),ival(2),ival(3))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  LOAD ip1 junit
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   80 continue
      call macparse(nwords,ipos+1,maclevel,index,2,0,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,2)
      call dataload(ival(1),ival(2))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> DATA EDIT page imin imax value
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   90 continue
      call macparse(nwords,ipos+1,maclevel,index,3,1,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,3)
      call mderefR(rval,vindexR,1)
      call datedit(ival(1),ival(2),ival(3),rval(1))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> DATA NORM type1/2 page1 page2
*                [1:improduct,2:RMS]
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  100 continue
      call macparse(nwords,ipos+1,maclevel,index,3,0,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,3)
      call datnorm(ival(1),ival(2),ival(3))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> DATA RMAX page imin imax
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  105 continue
      call macparse(nwords,ipos+1,maclevel,index,3,0,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,3)
      call datrmax(ival(1),ival(2),ival(3))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> DATA RSUM page imin imax
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  110 continue
      call macparse(nwords,ipos+1,maclevel,index,3,0,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,3)
      call datrsum(ival(1),ival(2),ival(3))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> DATA ADD cource targer factor
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  120 continue
      call macparse(nwords,ipos+1,maclevel,index,2,1,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,2)
      call mderefR(rval,vindexR,1)
      call datadd(ival(1),ival(2),rval(1))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> DATA SHAPE page type width
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  130 continue
      call macparse(nwords,ipos+1,maclevel,index,2,1,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,2)
      call mderefR(rval,vindexR,1)
      call datshape(ival(1),ival(2),rval(1))
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  HELP
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  140 continue
      call helplst(menu,help,nkeys)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*-->  RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  150 continue
      return

*---> TERMINATION
 4711 ircode = 0
      goto 9999
 1515 ircode = 0
      if(maclevel.eq.1) ircode = 1
      goto 9999
 1314 ircode = 2
      goto 9999
 1313 continue
      ircode = -1
 9999 continue
      if(ircode.ne.0.and.maclevel.eq.1) call rmcode
      if(ip.eq.0) goto 1
      END

***********************************************************************
*     DATA-MANAGER: control of spectra on pages
*                   (future implementation of TABLES)
*----------------------------------------------------------------------
      SUBROUTINE DATAPAGE(ip1,ip2)
*----------------------------------------------------------------------
*     SPECTRUM MANAGER: handles data pages system
*
*     ROOT-ROUTINE  DATAPAGE: show info on data pages
*
      IMPLICIT none
      INTEGER ip,ip1,ip2,junit,i,j,vindex(1),iform,isign,imin,imax,itype
      DOUBLE PRECISION ymin,ymax,sum,RMS,xRMS,offset,factor,inpro,
     .                 rval(2),value
      LOGICAL last,ok
      REAL rsum,xdummy
*
*vvvv spectrum data structure
      INTEGER mx1,mx2,n1,ipag,ifmt
      PARAMETER (mx1=1024,mx2=30)
      DOUBLE PRECISION dbuf(mx1/2,0:mx2)
      COMPLEX cbuf(mx1/2,0:mx2)
      REAL rbuf,xmin,xmax
      COMMON /X1BUF/ rbuf(mx1,0:mx2),xmin(0:mx2),xmax(0:mx2)
      COMMON /X1BUFI/ n1(0:mx2),ifmt(0:mx2),ipag
      EQUIVALENCE (rbuf,dbuf,cbuf)
*^^^^
      DATA RMS/0.D0/
      DATA iform /1/
*
      call chkrange(ip1,ip2,mx2,isign,ok)
      IF(.not.ok) THEN
          ip1 = 0
          ip2 = mx2
      ENDIF
      write(6,*)
     .'page ifmt n1   xmin        xmax        ymin        ymax      sum'
      DO 12 i = ip1, ip2
        ymin = 0.D0
        ymax = 0.D0
        sum = 0.0
        if(ifmt(i).eq.8) call cminmax(cbuf(1,i),n1(i),ymin,ymax)
        if(ifmt(i).eq.4) call rminmax(rbuf(1,i),n1(i),ymin,ymax)
        if(ifmt(i).eq.4) sum = rsum(rbuf(1,i),n1(i))
        if(ifmt(i).eq.-8)call dminmax(dbuf(1,i),n1(i),ymin,ymax)
        write(6,'(3(1X,I3),5(1X,G11.4))')
     .      i, ifmt(i), n1(i), xmin(i), xmax(i), ymin, ymax, sum
   12 CONTINUE
      RETURN
*----------------------------------------------------------------------
      ENTRY SPECSIM(ip)
*     simulate eseem spectrum on page ip
*----------------------------------------------------------------------
      call chkrange(ip,ip,mx2,isign,ok)
      if(.not.ok) goto 1013
      CALL simul(xmin(ip),xmax(ip),ifmt(ip),dbuf(1,ip),n1(ip),mx1/2,1)
      RETURN
*----------------------------------------------------------------------
      ENTRY EPR1SIM(ip)
*     simulate eseem spectrum on page ip
*----------------------------------------------------------------------
      call chkrange(ip,ip,mx2,isign,ok)
      if(.not.ok) goto 1013
      CALL EPR(xmin(ip),xmax(ip),ifmt(ip),dbuf(1,ip),n1(ip),mx1/2,1)
      RETURN
*----------------------------------------------------------------------
      ENTRY SPECPROC(ip1,ip2)
*     postprocess spectrum on page ip1; put result on page ip2
*----------------------------------------------------------------------
      call chkrange(ip1,ip2,mx2,isign,ok)
      if(.not.ok) goto 1013
      CALL pproc(xmin(ip1),xmax(ip1),ifmt(ip1),rbuf(1,ip1),n1(ip1),
     .           xmin(ip2),xmax(ip2),ifmt(ip2),rbuf(1,ip2),n1(ip2))
      RETURN
*----------------------------------------------------------------------
      ENTRY datnorm(itype,ip1,ip2)
*----------------------------------------------------------------------
      call chkrange(ip1,ip2,mx2,isign,ok)
      call chkint(itype,1,2,ok)
      if(.not.ok) goto 112
      RMS = 0.D0
      IF(ip1.eq.ip2.or.n1(ip1).eq.0.or.n1(ip2).eq.0) goto 112
*
      if(rsum(rbuf(1,ip1),n1(ip1)).eq.0.0) goto 112
      if(rsum(rbuf(1,ip2),n1(ip2)).eq.0.0) goto 112
      IF(itype.eq.1) THEN
         RMS = INPRO(rbuf(1,ip1),rbuf(1,ip2),n1(ip1))
      ELSE
         call xscale(rbuf(1,ip1),rbuf(1,ip2),n1(ip1),offset,factor)
         do 111 i = 1 , n1(ip1)
  111       RMS = RMS +((rbuf(i,ip1)-offset)*factor-rbuf(i,ip2))**2.0
            RMS = DSQRT(RMS)
      write(6,*) ' offset,factor,rms=',offset,factor,rms
      ENDIF
  112 continue
      rval(1) = RMS
      vindex(1) = 0
      call mpushR(rval,vindex,1)
      RETURN
*----------------------------------------------------------------------
      ENTRY datrmax(ip,ip1,ip2)
*     find maximum value in range ip1 ip2
*----------------------------------------------------------------------
      imin = ip1
      imax = ip2
      if(ip.lt.0.or.ip.gt.mx2) goto 1013
      if(imin.gt.imax) return
      if(imin.le.0) imin = 1
      if(imax.gt.mx1) imax = mx1
*
      call rminmax(rbuf(imin,ip),imax-imin+1,rval(2),rval(1))
      vindex(1) = 0
      call mpushR(rval,vindex,1)
      RETURN
*----------------------------------------------------------------------
      ENTRY datrsum(ip,ip1,ip2)
*     find maximum value in range ip1 ip2
*----------------------------------------------------------------------
      imin = ip1
      imax = ip2
      if(ip.lt.0.or.ip.gt.mx2) goto 1013
      if(imin.gt.imax) return
      if(imin.le.0) imin = 1
      if(imax.gt.mx1) imax = mx1
*
      rval(1) = rsum(rbuf(imin,ip),imax-imin+1)
      vindex(1) = 0
      call mpushR(rval,vindex,1)
      RETURN
*----------------------------------------------------------------------
      ENTRY DATACOPY(ip1,ip2)
*     COPY spectum on page ip1 to page p2
*----------------------------------------------------------------------
      call chkrange(ip1,ip2,mx2,isign,ok)
      if(.not.ok) goto 1013
      call rcopy(rbuf(1,ip1),rbuf(1,ip2),n1(ip1))
      n1(ip2) = n1(ip1)
      xmin(ip2) = xmin(ip1)
      xmax(ip2) = xmax(ip1)
      ifmt(ip2) = ifmt(ip1)
      RETURN
*----------------------------------------------------------------------
      ENTRY DATACLEAR(ip1,ip2)
*     CLEAR spectra from page ip1 to page ip2
*----------------------------------------------------------------------
      call chkrange(ip1,ip2,mx2,isign,ok)
      if(.not.ok) goto 1013
      DO 30 i = ip1, ip2 ,isign
      call rzero(rbuf(1,i),mx1)
      n1(i) = 0
      xmin(i) = 0
      xmax(i) = 0
      ifmt(i) = 4
   30 continue
      RETURN
*----------------------------------------------------------------------
      ENTRY datedit(ip,ip1,ip2,value)
*----------------------------------------------------------------------
      imin = ip1
      imax = ip2
      if(ip.lt.0.or.ip.gt.mx2) goto 1013
      if(imin.gt.imax) return
      if(imin.le.0) imin = 1
      if(imax.gt.mx1) imax = mx1
      do 33 i = imin, imax
   33 rbuf(i,ip) = value
      RETURN
*----------------------------------------------------------------------
      ENTRY datform(ip )
*----------------------------------------------------------------------
      iform = ip
      RETURN
*----------------------------------------------------------------------
      ENTRY DATADD(ip1,ip2,value)
*     ADD spectrum on page ip1 to spectrum on page p2 * factor (value)
*----------------------------------------------------------------------
      call chkrange(ip1,ip2,mx2,isign,ok)
      if(.not.ok) goto 1013
      n1(ip2) = n1(ip1)
      xmin(ip2) = xmin(ip1)
      xmax(ip2) = xmax(ip1)
      ifmt(ip2) = ifmt(ip1)
      do 44 i = 1 , n1(ip1)
  44     rbuf(i,ip2) = rbuf(i,ip2) + rbuf(i,ip1)*value
      RETURN
*----------------------------------------------------------------------
      ENTRY datshape(ip,itype,value)
*----------------------------------------------------------------------
      call chkrange(ip,ip,mx2,isign,ok)
      if(.not.ok) goto 1013
      if(ifmt(ip).ne.4) return
      value = value*n1(ip)/(xmax(ip)-xmin(ip))
      call shape(rbuf(1,ip),rbuf(1,0),n1(ip),itype,value)
      RETURN
*----------------------------------------------------------------------
      ENTRY DATASAVE(ip1,ip2,junit)
*     SAVE spectra from page ip1 to page ip2 on junit
*----------------------------------------------------------------------
      call chkrange(ip1,ip2,mx2,isign,ok)
      if(.not.ok) goto 1013
      write(6,*) ' saving in format:', iform
      IF(iform.eq.1) THEN
        DO 40 i = ip1 , ip2, isign
   40      call save1d(junit,rbuf(1,i),n1(i),xmax(i),xmin(i),ifmt(i))
      ELSE
       do 41 i = 1 , n1(ip1)
            write(junit,'(30(1X,G12.6))')
     .            xmin(ip1)+dble(i-1)*(xmax(ip1)-xmin(ip1))/n1(ip1),
     .            (rbuf(i,j), j = ip1, ip2, isign)
   41  continue
      ENDIF
      RETURN
*----------------------------------------------------------------------
      ENTRY DATALOAD(ip,junit)
*     LOAD spectra from junit starting on page ip
*----------------------------------------------------------------------
      IF(ip.lt.0.or.ip.gt.mx2) return
      IF(iform.eq.1) THEN
        i = ip -1
  60    i = i + 1
        call load1d(junit,rbuf(1,i),n1(i),xmax(i),xmin(i),ifmt(i),last)
        if(.not.last.and.i.le.mx2) goto 60
      ELSE
        i = 1
        read(junit,*,err=6013,end=66) xmin(ip), rbuf(i,ip)
  61    i = i+1
        if(i.gt.mx1) goto 66
        read(junit,*,err=6013,end=66) xdummy, rbuf(i,ip)
        goto 61
  66    n1(ip) = i-1
        xmax(ip) = xdummy
        ifmt(ip) = 4
        return
 6013   write(6,*) ' this file is not in 2-column format'
      ENDIF
      RETURN
*----------------------------------------------------------------------
      ENTRY DATAPLOT(ip1,ip2)
*     PLOT spectra from page ip1 to page ip2
*----------------------------------------------------------------------
      call chkrange(ip1,ip2,mx2,isign,ok)
      IF(.not.ok) goto 1013
*     call multplot(rbuf(1,ip1),n1(ip1),mx1,ip2-ip1+1,
*    .              xmin(ip1),xmax(ip2) )
      write(6,*) ' no DOS plotting implemented yet'
      RETURN
 1013 continue
      write(6,*) ' improper page number or range, mx2=',mx2
      END
*= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

      FUNCTION INPRO(ray1,ray2,n)
      DOUBLE PRECISION inpro,rnorm1,rnorm2
      REAL ray1(*),ray2(*)
      rnorm1 = 0.D0
      rnorm2 = 0.D0
      inpro = 0.D0
      DO 10 i = 1 , n
         rnorm1 = rnorm1 + ray1(i)**2
         rnorm2 = rnorm2 + ray2(i)**2
  10     inpro = inpro + ray1(i)*ray2(i)
      inpro = inpro /( dsqrt(rnorm1)*dsqrt(rnorm2) )
      END

      SUBROUTINE chkrange(i,j,max,isign,ok)
      LOGICAL ok
*     if(.not.ok) return
      ok = i.le.max.and.j.le.max.and.i.ge.0.and.j.ge.0
      if(i.le.j) isign = 1
      if(i.gt.j) isign = -1
      end

*----------------------------------------------------------------------
      SUBROUTINE save1d(junit,rbuf,n1,xmin,xmax,ifmt)
*----------------------------------------------------------------------
      IMPLICIT none
      REAL rbuf(*),xmin,xmax
      INTEGER junit,n1,ifmt,i
      write(junit,'(2(1X,I3),2(1X,G12.5))') n1, ifmt, xmin, xmax
      write(junit,'(10Z8)') (rbuf(i), i = 1 , n1)
      END
*----------------------------------------------------------------------
      SUBROUTINE load1d(junit,rbuf,n1,xmin,xmax,ifmt,fail)
*----------------------------------------------------------------------
      IMPLICIT none
      REAL rbuf(*),xmin,xmax
      INTEGER junit,n1,ifmt,i
      LOGICAL fail
      read(junit,*,end=1000,err=1313) n1, ifmt, xmin, xmax
      read(junit,'(10Z8)',end=1013,err=1313) (rbuf(i), i = 1 , n1)
      fail = .false.
      return
 1000 fail = .true.
      return
 1013 fail = .true.
      write(6,*) ' unexpected EOF during load1d'
      return
 1313 fail = .true.
      write(6,*) ' I/O error'
      return
      END
*--------------------------------------------------------------------
      SUBROUTINE shape(y,y0,n,itype,width)
*--------------------------------------------------------------------
      IMPLICIT none
      INTEGER max,n
      PARAMETER (max=100)
      DOUBLE PRECISION width
      REAL y(n),y0(n),gauss,gderiv,lorentz,lderiv,xfrac,rwidth
      REAL stalin(-max:max),widthfact(4),xnorm(4),rnorm
      INTEGER itype,isteps,itypold,iexp(4),
     .        i,ioff,is,ilo,ihi
      DATA widthfact /1.5,2.0,11.0,10.0/
      DATA iexp /-1,-2,-1,-2/
      DATA xnorm /0.939437,3.191538,0.6366198,0.5513289/
      DATA itypold  /0/
      DATA rwidth  /0.0/
*-STATEMENT FUNCTIONS: input postions in fraction of linewidth
      gauss(xfrac) = EXP(-2.7725887224E0 * xfrac**2)
      gderiv(xfrac) = -xfrac * EXP(-2.E0*xfrac**2)
      lorentz(xfrac) = (1/4.0)/(1/4.0 + xfrac**2 )
      lderiv(xfrac) = -xfrac/(0.75+xfrac**2)**2
*-------------                           gauss(1.5) = 0.002..
*                                      lorentz(11)= 0.002..
      IF(itype.le.0.or.itype.gt.4) stop ' illegal call to SHAPE'
      isteps = widthfact(itype)*DABS(width) + 1
      IF(isteps.gt.max) isteps = max
      IF(itype.eq.itypold.and.width.eq.rwidth) goto 90
      itypold = itype
      rwidth = width
      rnorm = xnorm(itype)*dabs(width)**(iexp(itype))
*
      IF (itype.EQ.1) THEN
*     GAUSSIAN
      do 10 i = -isteps, isteps
   10 stalin(i) = gauss(FLOAT(i)/rwidth)
*
      ELSE IF (itype.EQ.2) THEN
*     GAUSSIAN DERIVATIVE
      do 20 i = -isteps, isteps
   20 stalin(i) = gderiv(FLOAT(i)/rwidth)
*
      ELSE IF (itype.eq.3) then
*     LORENTZIAN
      do 30 i = -isteps, isteps
   30 stalin(i) = lorentz(FLOAT(i)/rwidth)
*
      ELSE IF (itype.eq.4) then
*     LORENTZIAN DERIVATIVE
      do 40 i = -isteps, isteps
   40 stalin(i) = lderiv(FLOAT(i)/rwidth)

      ELSE
          STOP ' ILLEGAL CALL TO SHAPE '
      ENDIF
*
*---  CONVOLUTE LINESHAPE
*
   90 continue
      do 95 i = 1 , n
      y0(i) = y(i)
   95 y(i) = 0.D0
*
      do 200 i = 1 , n
      IF(y0(i).eq.0.0) GOTO 200
      ilo = i - isteps
      ihi = i + isteps
      ioff =  - ilo - isteps
      if(ilo.lt.1) ilo = 1
      if(ihi.gt.n) ihi = n
      do 100 is = ilo , ihi
      y(is) = y(is) + y0(i)*rnorm*stalin(is+ioff)
  100 CONTINUE
  200 CONTINUE
      END
