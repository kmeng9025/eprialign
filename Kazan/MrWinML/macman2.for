*     MACRO MANAGER
*-----------------------------------------------------------------------
      SUBROUTINE clearmac
*-----------------------------------------------------------------------
*     Clear macro structure
*
*     GENERAL IDEA: During interpretation of a interactive session
*     using commands and arguments the commands are decoded into an
*     index which is added in the "code"-list; the arguments are decoded
*     and accomodated in the "reals/ints/string"-list.
*
*     For every "code" (basic command) the entry point in the argument
*     list is stored in the "iargp"- array.
*
*     The following counters are in effect:
*     iap : argument pointer
*     isp : stack pointer
*     ipc : program counter (pointer in code-list)
*
*     mxmac: maximum number of macro-lines
*     mxarg: maximum number of argument units (8 bytes)
*     mxvar: maximum number of variables
*
*     DATA SEGMENT ORGANIZATION:
*
*      ______________________________________________________________
*     |   char*1  dtype  (R:real,I:integer,S:string)                 |
*      ______________________________________________________________
*     |   integer locvar  (index of variable)                        |
*      ________________ _____________________________________________
*     |________________|_____________________________________________|
*        <---   stack  ^     --->    argument space  --->            ^
*                     ibase[100]                            mxarg[1000]
*
*     REFERENCE STRUCTURE:
*
*            ipc               iap   isp
*             |                 |     |
*             v                 v     v
*       -----------       ------------------
*      | mcode     |     | C/I/R- arguments |
*       -----------       ------------------         --------------
*      | iargp     |---> |  dtype           |       |  Rvar        |
*       -----------       ------------------         --------------
*                        |  locvar          | ----> |  Ivar        |
*                         ------------------         --------------
*
*
*vvvvv
*===> parameters MACRO STRUCTURE
      IMPLICIT none
      INTEGER mxmac,mxarg,mxvar,mxloop
      PARAMETER (mxmac=500,mxarg=1000,mxvar=20,mxloop=10)
*MACRO STORAGE SPACE
      CHARACTER*8 chars(mxarg)
      CHARACTER*1 dtype(mxarg)
      DOUBLE PRECISION reals(mxarg),Rvar(mxvar)
      INTEGER Ivar(mxvar),ints(mxarg*2),locvar(mxarg),mcode(mxmac),
     .        iargp(mxmac),ibase,iap,ipc,isp,nlines,
     .        Lstk(mxloop),Lpnt
      EQUIVALENCE (chars,reals,ints)
*vv
      INTEGER merror
      COMMON /macro/ merror
*^^
*temp parameters and ENTRY arguments
*-> real / int argument exchange:
      INTEGER i,j,k,Ival(*),nval,vindex(*),icode,junit
***   INTEGER Iv,idx
      DOUBLE PRECISION Rval(*),error,err
***   DOUBLE PRECISION Rv
*     CHARACTER*1 vtype
*-> string argument exchange
      INTEGER nchunk,nleft,lenstring
      INTEGER ifirst,ilast
      CHARACTER*(*) string
*-> position base for argument arrays
      DATA ibase /100/
*
      DATA nlines /0/
      DATA error  /0.D0/

*---> INIT
      call izero(ints,mxarg*2)
      call izero(locvar,mxarg)
      call izero(mcode  ,mxmac)
      call izero(iargp  ,mxmac)
      call izero(Lstk   ,mxloop)
      call izero(Ivar   ,mxvar)
      call dzero(Rvar   ,mxvar)
      nlines = 0
* - - - - - - - - - -
      ENTRY resetmac
* - - - - - - - - - -
      iap = ibase
      ipc = 1
      isp = ibase
      Lpnt = 0
      merror = 0
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mputcode(icode)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      IF(ipc.lt.mxmac) THEN
        iargp(ipc) = iap
        mcode(ipc) = icode
        ipc = ipc + 1
        nlines = nlines + 1
      ELSE
        merror = 1
        goto 1313
      ENDIF
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mqlines(nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> query: number of macrolines
      nval = nlines
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY rmcode
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> back up code pointer one position [convenient for error correction
      IF(ipc.gt.1) THEN
         ipc = ipc - 1
         iap = iargp(ipc)
         nlines = nlines -1
      ENDIF
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mgetcode(icode)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if(iargp(ipc).ne.iap) merror = 3
      if(ipc.gt.nlines) merror = 4
      if(ipc.gt.mxmac) merror = 1
      if(merror.ne.0) goto 1313
      if(nlines.eq.0) then
         icode = -1
      else
         icode = mcode(ipc)
         ipc = ipc + 1
      endif
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mputreal(Rval,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      IF(iap+nval.lt.mxarg) THEN
        DO 10 i = 1 , nval
          reals(iap) = Rval(i)
          locvar(iap) = vindex(i)
          dtype(iap) = 'R'
          iap = iap + 1
  10    continue
      ELSE
        merror = 2
        goto 1313
      ENDIF
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mgetreal(Rval,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> read back stored argument (no type checking)
      IF(iap+nval.lt.mxarg) THEN
        DO 20 i = 1 , nval
          Rval(i) = reals(iap)
          vindex(i) = locvar(iap)
          iap = iap + 1
  20    continue
      ELSE
        merror = 2
        goto 1313
      ENDIF
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mderefR(Rval,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> Dereference variables (fill in value)
      DO 25 i = 1 , nval
      if(vindex(i).le.0.or.vindex(i).gt.mxvar) goto 25
      Rval(i) = Rvar(vindex(i))
   25 continue
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mrefR(Rval,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      DO 27 i = 1 , nval
      if(vindex(i).le.0.or.vindex(i).gt.mxvar) goto 27
      Rvar(vindex(i)) = Rval(i)
   27 continue
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mputint(Ival,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      IF(iap+nval.lt.mxarg) THEN
        DO 30 i = 1 , nval
          ints(2*iap) = Ival(i)
          locvar(iap) = vindex(i)
          dtype(iap) = 'I'
          iap = iap + 1
  30    continue
      ELSE
        merror = 2
        goto 1313
      ENDIF
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mgetint(Ival,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      IF(iap+nval.lt.mxarg) THEN
        DO 40 i = 1 , nval
          Ival(i) = ints(2*iap)
          vindex(i) = locvar(iap)
          iap = iap + 1
  40    continue
      ELSE
        merror = 2
        goto 1313
      ENDIF
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mderefI(Ival,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      DO 45 i = 1 , nval
      if(vindex(i).le.0.or.vindex(i).gt.mxvar) goto 45
      Ival(i) = Ivar(vindex(i))
   45 continue
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mrefI(Ival,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      DO 50 i = 1 , nval
      if(vindex(i).le.0.or.vindex(i).gt.mxvar) goto 50
      Ivar(vindex(i)) = Ival(i)
   50 continue
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mpushR(Rval,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> push constant or variable on stack
      IF(isp-nval.gt.0) THEN
        DO 70 i = 1 , nval
          isp = isp - 1
          reals(isp) = Rval(i)
          locvar(isp) = vindex(i)
          dtype(isp) = 'R'
  70    CONTINUE
      ELSE
        merror = 7
        goto 1313
      ENDIF
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mpopR(Rval,vindex,nval)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> pop value on stack into Rval
      IF(isp+nval.le.ibase) THEN
        DO 80 i = 1 , nval
          Rval(i) = reals(isp)
          vindex(i) = locvar(isp)
          isp = isp + 1
  80    CONTINUE
      ELSE
        merror = 6
        goto 1313
      ENDIF
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mpoperr
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      IF(isp+1.le.ibase) THEN
        if(locvar(isp).ne.0) reals(isp) = Rvar(locvar(isp))
        error = reals(isp)
        isp = isp + 1
      ELSE
        merror = 6
        goto 1313
      ENDIF
      RETURN
**- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*      ENTRY mpush(Rv,Iv,vtype,idx)
**- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
**---> push constant or variable on stack
*        if(isp.le.0) stop ' STACK OVERFLOW'
*        IF(idx.lt.0.or.idx.gt.mxvar) stop ' illegal variable index'
*        isp = isp - 1
*        dtype(isp) = vtype
*        IF(index.eq.0) THEN
*           if(vtype.eq.'R') reals(isp) =  Rv
*           if(vtype.eq.'I') ints(2*isp) = Iv
*        ELSE
*                            locvar(isp) = index
*        ENDIF
*      RETURN
**- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*      ENTRY mpop(idx,vtype)
**- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
**---> pop value on stack into variables index
*      if(isp.ge.ibase) stop ' STACK UNDERFLOW'
*      IF(idx.eq.0.or.idx.gt.mxvar) goto 85
*      if(locvar(isp).ne.0) goto 81
**
*      if(dtype(isp).eq.vtype) then
*         if(vtype.eq.'R') Rvar(index) = reals(isp)
*         if(vtype.eq.'I') Ivar(index) = ints (2*isp)
*      else if (vtype.eq.'R') then
*                          Rvar(index) = dble(ints(2*isp))
*      else
*           ierr = 1
*           goto 1313
*      endif
*      goto 85
*  81  continue
*      if(dtype(isp).eq.vtype) then
*         if(vtype.eq.'R') Rvar(idx) = Rvar(locvar(isp))
*         if(vtype.eq.'I') Ivar(idx) = Ivar(locvar(isp))
*      else if (vtype.eq.'R') then
*                          Rvar(idx) = dble(Ivar(locvar(isp)))
*      else
*           ierr = 1
*           goto 1313
*      endif
*   85 isp = isp + 1
*      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY msetvarR(Rval,ifirst,ilast)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      DO 90 i = 1 , ilast-ifirst+1
  90  Rvar(i+ifirst-1) = Rval(i)
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY msetvarI(Ival,ifirst,ilast)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      DO 100 i = 1 , ilast-ifirst+1
 100  Ivar(i+ifirst-1) = Ival(i)
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY maddvarR(Rval,ifirst,ilast)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      DO 110 i = 1 , ilast-ifirst+1
 110  Rvar(i+ifirst-1) = Rvar(i+ifirst-1) + Rval(i)
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY maddvarI(Ival,ifirst,ilast)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      DO 120 i = 1 , ilast-ifirst+1
 120  Ivar(i+ifirst-1) = Ivar(i+ifirst-1) + Ival(i)
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY geterror(err)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      err = error
      RETURN

*=================== STRING OPERATIONS =================================

*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY mputstring(string)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      lenstring = len(string)
      ints(2*iap) = lenstring
      iap = iap + 1
      if(iap.gt.mxarg) stop ' macro argument segment exceeded'
      nchunk = lenstring / 8
      nleft  = MOD(lenstring,8)
      DO 500 i = 1 , nchunk
         j = (i-1)*8+1
         k = j + 7
         chars(iap) = string(j:k)
         dtype(iap) = 'S'
         iap = iap + 1
         if(iap.gt.mxarg) stop ' macro argument segment exceeded'
 500  continue
      j = nchunk*8+1
      k = j + nleft-1
      chars(iap) = string(j:k)
      dtype(iap) = 'S'
      iap = iap + 1
      if(iap.gt.mxarg) stop ' macro argument segment exceeded'
      RETURN

*========================== CONTROL OPERATIONS ========================

*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY DO(vindex,Ival)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Lpnt = Lpnt + 2
      Lstk(Lpnt) = ipc
      Lstk(Lpnt+1) = vindex(1)
      Ivar(vindex(1)) = Ival(1)
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY LOOP
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      i = Lstk(Lpnt+1)
      Ivar(i) = Ivar(i) - 1
      IF(Ivar(i).le.0) THEN
         Lpnt = Lpnt - 2
      ELSE
         ipc = Lstk(Lpnt)
         iap = iargp(ipc)
      ENDIF
      RETURN

*========================== MISCELLANIOUS ========================
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY MLIST(junit)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      write(junit,'('' pointers: ipc='',I3,'' iap='',I3,'' isp='',I3,
     .              '' Lpnt='',I3)') ipc,iap,isp,Lpnt
      write(junit,'('' mcode:'',20I3)') (mcode(i), i = 1 , nlines)
      write(junit,'('' iargp:'',20I3)') (iargp(i), i = 1 , nlines)
*
      write(junit,
     .'(''   reals     ints   chars    dtype    loc'')')
      do 600 i = ibase , iap
      write(junit,'(1X,G12.5,1X,I6,1X,A8,1X,A1,1X,I4)')
     .   reals(i),ints(2*i),chars(i),dtype(i),locvar(i)
  600 continue
      write(junit,*) ' STACK:'
      do 610 i = ibase , isp, -1
      write(junit,'(1X,G12.5,1X,I6,1X,A8,1X,A1,1X,I4)')
     .   reals(i),ints(2*i),chars(i),dtype(i),locvar(i)
  610 continue
      write(junit,'('' RVAR:'')')
      write(junit,'(8(1X,G9.3))') (rvar(i), i = 1 , mxvar)
      write(junit,'('' IVAR:'')')
      write(junit,'(8(1X,I9  ))') (ivar(i), i = 1 , mxvar)
      RETURN
*+++++++++++++++++++++++
*---> ERROR handling
 1313 continue
      call mmessag(6,merror)
      END

*--------------------------------------------------------------------
      SUBROUTINE mmessag(junit,icode)
*--------------------------------------------------------------------
      PARAMETER (nm=7)
      CHARACTER*60 mess(nm)
      DATA mess
     ./'MACRO CODE SEGMENT EXCEEDED                                 ',
     . 'MACRO ARGUMENT SEGMENT EXCEEDED                             ',
     . 'MACRO CODE/ARGUMENT MISALLIGNMENT                           ',
     . 'MACRO CODE POINTER OUT OF RANGE                             ',
     . 'MACRO ERROR: TYPE CONFLICT                                  ',
     . 'MACRO STACK UNDERFLOW                                       ',
     . 'MACRO STACK OVERFLOW                                        '/
      if(icode.lt.nm.and.icode.gt.0)
     .   write(junit,'(2X,A)') mess(icode)
      END
*======================================================================
*     I/O routines
*     addendum to usrin.for (to be implemented as one package ...)
*======================================================================
      SUBROUTINE getiray(ipos,iray,vindex,n,ok)
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER ipos,iray(*),vindex(*),i,n,nr
      CHARACTER*72 word
      LOGICAL ok
      if(.not.ok.or.n.le.0) return
      DO 1 i = 1 , n
         call gtword(ipos+i-1,word,nr)
         if(nr.le.0) goto 13
         IF(word(1:1).eq.'#') THEN
            read(word(2:72),'(I6)',err=13) vindex(i)
            iray(i) = 0
         ELSE
            read(word,'(I6)',err=13) iray(i)
            vindex(i) = 0
         ENDIF
    1 CONTINUE
      return
   13 ok = .false.
      write(6,*) ' illegal integer or variable'
      END

*----------------------------------------------------------------------
      SUBROUTINE getrray(ipos,rray,vindex,n,ok)
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER ipos,i,n,vindex(*),nr
      DOUBLE PRECISION rray(*)
      CHARACTER*72 word
      LOGICAL ok
      IF(.not.ok.or.n.le.0) return
      DO 1 i = 1 , n
         call gtword(ipos+i-1,word,nr)
         if(nr.le.0) goto 13
         IF(word(1:1).eq.'$') THEN
            read(word(2:72),'(I6)',err=13) vindex(i)
            rray(i) = 0.D0
         ELSE
            call gtreal(ipos+i-1,rray(i),ok)
            if(.not.ok) goto 13
            vindex(i) = 0
         ENDIF
    1 CONTINUE
      return
   13 ok = .false.
      write(6,*) ' illegal real or variable'
      END

      SUBROUTINE gtfname(ipos,nwords,fname,len)
      IMPLICIT none
      INTEGER ipos,nwords,nr,len,iw
      CHARACTER*64 fname,word
*--init
      fname = ' '
      len = 1
*
      if(ipos.gt.nwords) return
      call gtword(ipos,fname,len)
      if(ipos.eq.nwords) return
*
      DO 10 iw = ipos+1 , nwords
      call gtword(iw,word,nr)
      fname = fname(1:len) //' '// word(1:nr)
      len = len + nr + 1
   10 continue
      end

*----------------------------------------------------------------------
      SUBROUTINE macparse(nwords,ipos,maclevel,index,nint,nreal,
     .                    ival,rval,vindexI,vindexR,narg)
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER nwords,ipos,maclevel,index,nint,nreal,narg
      INTEGER ival(*),vindexR(*),vindexI(*)
      DOUBLE PRECISION rval(*)
      LOGICAL ok
*
      IF(maclevel.le.1) THEN
         IF(nwords-ipos+1.le.0) goto 1414
         if(nwords-ipos+1.lt.nint+nreal) goto 1313
         ok = .true.
         call getiray(ipos     ,ival,vindexI,nint ,ok)
         call getrray(ipos+nint,rval,vindexR,nreal,ok)
         if(.not.ok) goto 1313
      ENDIF
      IF(maclevel.eq.1) THEN
*        call mputcode(0)
         call mputcode(index)
         call mputint (ival,vindexI,nint )
         call mputreal(rval,vindexR,nreal)
      ELSE IF(maclevel.ge.2) THEN
         call mgetint (ival,vindexI,nint )
*        call mderefI (ival,vindexI,nint )
         call mgetreal(rval,vindexR,nreal)
*        call mderefR (rval,vindexR,nreal)
      ENDIF

 4711 narg = 1
      return
 1414 continue
      narg = 0
      return
 1313 narg = -1
      return
      END

*----------------------------------------------------------------------
      SUBROUTINE chkreal(rval,rmin,rmax,ok)
*----------------------------------------------------------------------
      DOUBLE PRECISION rval,rmin,rmax
      LOGICAL ok
      if(.not.ok) return
      if(rval.lt.rmin.or.rval.gt.rmax) ok = .false.
      if(.not.ok) write(6,*) ' real out of bounds'
      end
*----------------------------------------------------------------------
      SUBROUTINE chkint(ival,imin,imax,ok)
*----------------------------------------------------------------------
      LOGICAL ok
      if(.not.ok) return
      if(ival.lt.imin.or.ival.gt.imax) ok = .false.
      if(.not.ok) write(6,*) ' integer out of bounds'
      end

*----------------------------------------------------------------------
      SUBROUTINE MACIN(nwords,maclevel)
*----------------------------------------------------------------------
*     A command 'FILTER' between the user-menu's and USERIN.
*     &macro directives are recognized and decoded
*     Only non-macro directives are passed to the calling program
*
      IMPLICIT none
      INTEGER nwords,maclevel
*---  local parameters
      INTEGER nkeys,index,noccur,ircode
      PARAMETER (nkeys=22)
      LOGICAL ok
*
      INTEGER mxval
      PARAMETER (mxval=20)
      INTEGER ival(mxval),vindex(mxval),vindex2(mxval),n1(1),n2(1),i,
     .       nval,merror,ipos
      DOUBLE PRECISION rval(mxval),rval2(mxval)
      LOGICAL xset,xadd,xpush,xpop
*
      CHARACTER*10 menu(nkeys)
      COMMON /macro/ merror
      DATA menu /'&MACRO    ','&END      ','&DO       ','&LOOP     ',
     .           '&SETVAR$  ','&ADDVAR$  ','&PUSH$    ','&POP$     ',
     .           '&SETVAR#  ','&ADDVAR#  ','&X$       ','&/$       ',
     .           '&+$       ','&-$       ','&TEST     ','&EXEC     ',
     .           '&HELP     ','&SQRT$    ','&XX$      ','&LIST     ',
     .           '&ERR      ','&SWAP$    '/
      DATA ipos /1/
****  DATA ircode /0/
*********************************************************************
*
*     maclevel: 0 = inactive (direct interpretation of input
*               1 = macro definition (no execution)
*               2 = 'dry' testrun of macro
*               3 = full macro execution
*
**********************
*========>>>>>  start user monitor loop
***   maclevel = mlvl
    1 continue
*
      ok = .true.
      xset = .false.
      xadd = .false.
      xpush= .false.
      xpop = .false.
*-------------> interpretation/definition level
      IF(maclevel.le.1) THEN
        CALL userin(nwords)
        call recog(1,menu,nkeys,index,noccur)
        IF(noccur.eq.0) THEN
           if(maclevel.eq.1) call mputcode(0)
           return
        ELSE IF(noccur.gt.1) THEN
           !write(6,*) ' AMBIGUOUS MACRO-DIRECTIVE'
           goto 1
        ENDIF
      ELSE
*-------------> execute level
      call mgetcode(index)
      if(merror.ne.0) then
           !write(6,*) ' MACRO EXECUTION ABORTED, error-code=', merror
           maclevel = 0
           goto 1
      endif
      if(index.eq.0) return
      ENDIF
*===>>> COMMAND DISPATCHER
      goto(10,20,30,40,50,60,70,80,100,110,120,125,130,135,140,
     .     150,160,170,175,180,190,200)  index
      !write(6,*) ' illegal code in MACMON driver, index=', index
      STOP
*
*===========>>>>>>>>>> execute commands <<<<<<<<<<<<<<==========
*
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&MACRO' : start macro interpretation
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   10 continue
      !write(6,*) ' begin macro interpretation'
      call clearmac
      maclevel = 1
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&END' End Macro interpretation
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   20 continue
      !write(6,*) ' end macro interpretation/execution'
      goto(21,22,22) maclevel
        !write(6,*) ' no macro active; command ignored'
        goto 4711
   21 continue
        call mputcode(index)
        maclevel = 0
        goto 4711
   22 continue
        maclevel = 0
        goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&DO #n count' start do-loop, use index var #n as counter
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   30 continue
      goto(31,32,32) maclevel
        !write(6,*) ' no macro active; command ignored'
        goto 4711
   31 continue
        call getiray(ipos+1,ival,vindex,2,ok)
        if(.not.ok) goto 1313
        if(vindex(1).eq.0) goto 1313
        if(vindex(2).ne.0) goto 1313
        call mputcode(index)
        call mputint(ival,vindex,2)
        goto 4711
   32 continue
        call mgetint(ival,vindex,2)
        call do(vindex(1),ival(2))
        goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&LOOP' set loopback point
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   40 continue
      goto(41,42,42) maclevel
        !write(6,*) ' no macro active; command ignored'
        goto 4711
   41 continue
        call mputcode(index)
        goto 4711
   42 continue
        call loop
        goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - -
*=====> '&SETVAR$' val1 val2 .... SET real variables
*NOTE: range item [n,m]/[n] to be implemented
*- - - - - - - - - - - - - - - - - - - - - - - - - -
   50 continue
      xset = .true.
      goto 81
*- - - - - - - - - - - - - - - - - - - - - - - - - -
*=====> '&ADDVAR$' val1 val2 .... ADD real variables
*- - - - - - - - - - - - - - - - - - - - - - - - - -
   60 continue
      xadd = .true.
      goto 81
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===>  '&PUSH$' push a series of real values/variable on the stack
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   70 continue
      xpush= .true.
      goto 81
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===>  '&POP$' pop real values into variables  ($1 $2 $3 ... )
*      non variables are ignored (act as dummy variables)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   80 continue
      xpop = .true.
*===================> combined set/add/push/pop code <===========
   81 continue
*---> INPUT
      if(maclevel.le.1) then
         nval = nwords-ipos
         call getrray(ipos+1,rval,vindex,nval,ok)
         if(.not.ok) goto 1313
      endif
*---> DEFINITION
      if(maclevel.eq.1) then
         call mputcode(index)
         n1(1) = nval
         n2(1) = 0
         call mputint(n1,n2,1)
         call mputreal(rval,vindex,nval)
         goto 4711
*---> EXECUTION
      else if(maclevel.ge.2) then
         call mgetint(n1,n2,1)
         nval = n1(1)
         call mgetreal(rval,vindex,nval)
      endif
      if(xset) then
         call mderefR(rval,vindex,nval)
         call msetvarR(rval,1,nval)
      else if(xadd) then
         call mderefR(rval,vindex,nval)
         call maddvarR(rval,1,nval)
      else if(xpush) then
         call mpushR(rval,vindex,nval)
      else if(xpop) then
         call mpopR(rval2,vindex2,nval)
         call mderefR(rval2,vindex2,nval)
         call mrefR(rval2,vindex,nval)
      else
         stop ' illegal set/add/push/pop/ operation '
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - -
*=====> '&SETVAR#' val1 val2 .... SET int variables
*- - - - - - - - - - - - - - - - - - - - - - - - - -
  100 continue
      xset = .true.
      goto 111
*- - - - - - - - - - - - - - - - - - - - - - - - - -
*=====> '&ADDVAR#' val1 val2 .... ADD int variables
*- - - - - - - - - - - - - - - - - - - - - - - - - -
  110 continue
      xadd = .true.
*
  111 continue
*---> INPUT
      if(maclevel.le.1) then
         nval = nwords-ipos
         call getiray(ipos+1,ival,vindex,nval,ok)
         if(.not.ok) goto 1313
      endif
*---> DEFINITION
      if(maclevel.eq.1) then
         call mputcode(index)
         n1(1) = nval
         n2(1) = 0
         call mputint(n1,n2,1)
         call mputint (ival,vindex,nval)
         goto 4711
*---> EXECUTION
      else if(maclevel.ge.2) then
         call mgetint(n1,n2,1)
         nval = n1(1)
         call mgetint (ival,vindex,nval)
      endif
      call mderefI(ival,vindex,nval)
      if(xset) then
         call msetvarI(ival,1,nval)
      else if(xadd) then
         call maddvarI(ival,1,nval)
      else
         stop ' illegal set/add # operation'
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&X$' multiply top two items on stack and replace top with
*           result
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  120 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call mpopR(rval,vindex,2)
         call mderefR(rval,vindex,2)
         rval(1) = rval(2) * rval(1)
         vindex(1) = 0
         call mpushR(rval,vindex,1)
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&/$' DIVIDE top two items on stack and replace top with
*           result
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  125 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call mpopR(rval,vindex,2)
         call mderefR(rval,vindex,2)
         if(dabs(rval(1)).lt.1.D-20) rval(1) = 1.D-20
         rval(1) = rval(2) / rval(1)
         vindex(1) = 0
         call mpushR(rval,vindex,1)
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&+$' add top two items on stack and replace top with
*           result
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  130 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call mpopR(rval,vindex,2)
         call mderefR(rval,vindex,2)
         rval(1) = rval(2) + rval(1)
         vindex(1) = 0
         call mpushR(rval,vindex,1)
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&-$' substract top two items on stack and replace top with
*           result
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  135 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call mpopR(rval,vindex,2)
         call mderefR(rval,vindex,2)
         rval(1) = rval(2) - rval(1)
         vindex(1) = 0
         call mpushR(rval,vindex,1)
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> &TEST : test mode for macro execution
*             commands which invoke complicated calculations
*             (e.g. spectral simulations) are not executed
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  140 continue
      goto (141,142,142) maclevel
      call mqlines(nval)
      if(nval.le.0) then
        !write(6,*) ' no macro defined'
        goto 4711
      endif
      maclevel = 2
      call resetmac
      goto 4711
	!write(6,*) ' ignored during macro definition'
  141    goto 4711
  142 continue
      STOP ' BUG! -&TEST cannot occur during macro execution !!!'
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> &EXEC : macro execution
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  150 continue
      goto (151,152,152) maclevel
      call mqlines(nval)
      if(nval.le.0) then
        !write(6,*) ' no macro defined'
        goto 4711
      endif
      maclevel = 3
      call resetmac
      goto 4711
   !write(6,*) ' ignored during macro definition'
  151    goto 4711
  152 continue
      STOP ' BUG! -&EXEC cannot occur during macro execution !!!'

*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> &HELP
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  160 continue
      call comlist(menu,nkeys)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&SQRT$' take sqrt of item on top of stack (and replace)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  170 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call mpopR(rval,vindex,1)
         call mderefR(rval,vindex,1)
         rval(1) = dsqrt(rval(1))
         vindex(1) = 0
         call mpushR(rval,vindex,1)
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&XX$' take square of item on top of stack (and replace)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  175 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call mpopR(rval,vindex,1)
         call mderefR(rval,vindex,1)
         rval(1) = rval(1)**2
         vindex(1) = 0
         call mpushR(rval,vindex,1)
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> &LIST (macro arrays)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  180 continue
      call mlist(6)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> &ERR  pop stack into err-register
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  190 continue
      if(maclevel.eq.1) call mputcode(index)
      call mpoperr
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*===> '&SWAP$' SWAP top two items on stack
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  200 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         call mpopR(rval,vindex,2)
         call mpushR(rval,vindex,2)
      endif
      goto 4711

* NORMAL TERMINATION
 4711 continue
*     ircode = 0
      goto 1
* ERROR TERMINATION
 1313 continue
      call message(-1)
      goto 1
 1515 continue
*     ircode = 0
      if(maclevel.eq.1) call message(1)
      goto 1
      END

*--------------------------------------------------------------------
      SUBROUTINE message(ircode)
*--------------------------------------------------------------------
      PARAMETER (nm=3)
      CHARACTER*60 messag(nm)
      DATA messag
     ./' WARNING: command not active during macro execution         ',
     . 'I/O ERROR                                                   ',
     . '.........x.........x.........x.........x.........x.........x'/
      !if(ircode.lt.0) write(6,*) ' INPUT not recognized'
      !if(ircode.lt.nm.and.ircode.gt.0)
     .!   write(6,'(2X,A)') messag(ircode)
      END
*----------------------------------------------------------------
      SUBROUTINE comlist(menu,nkeys)
*----------------------------------------------------------------
      CHARACTER*(*) menu(nkeys)
      write(6,'(2X,A)') ( menu(i), i = 1 , nkeys )
      END

*----------------------------------------------------------------
      SUBROUTINE helplst(menu,help,nkeys)
*----------------------------------------------------------------
      CHARACTER*(10) menu(nkeys)
      CHARACTER*(60) help(nkeys)
      write(6,'(2X,A)') ( menu(i)//help(i), i = 1 , nkeys )
      END
