*-----------------------------------------------------------------------
      SUBROUTINE FCN (NPVAR,Y,FUNC,X,IFLAG)
*-----------------------------------------------------------------------
*     TRIVIAL INTERFACE BETWEEN THE PROGRAMS MINUIT AND MR2
*     ONE MAY CONSIDER WRITING ONE'S OWN FCN IF A TRANSFORMATION
*     OF THE MINIMIZING PARAMETERS (X) IS DESIRABLE.
 
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/PAREXT/ U(50)       ,W(50)      ,WERR(50)   ,MAXEXT     ,NU
      REAL*8 Y(NU),X(NU)
*        Y ARE 1ST DERIVATIVES; X ARE OPTIMIZING PARAMETERS.
      CALL mr2(NU,FUNC,X,IFLAG)
      END
 
*-----------------------------------------------------------------------
      SUBROUTINE mr2 (nvar,error,var,iflag)
*-----------------------------------------------------------------------
*     INTERFACING ROUTINE BETWEEN OPTIMIZING PROGRAM AND MR2-SYSTEM
      IMPLICIT none
      INTEGER nvar,iflag
      DOUBLE PRECISION error,var(*)
*---> first call
      if(iflag.eq.1) then
         call clearmac
*             read input
         call mainmon(0)
         call resetmac
*             execute optimizer macro
         call msetvarR(var,1,nvar)
         call mainmon(3)
      else
         call resetmac
*             execute optimizer macro
         call msetvarR(var,1,nvar)
         call mainmon(3)
      endif
      call geterror(error)
*---> last call:   read last part of input
      if(iflag.eq.3) call mainmon(0)
      END
*
*----------------------------------------------------------------------
      SUBROUTINE mainmon(maclevel)
*----------------------------------------------------------------------
      IMPLICIT none
      INTEGER nkeys,mxval,index,noccur,ircode,nwords,ipos,maclevel
      PARAMETER (nkeys=11,mxval=20)
      LOGICAL ok
*---  global parameter(s)
      INTEGER merror
      COMMON /macro/ merror
*---  local parameters
      INTEGER ival(mxval),vindexR(mxval),vindexI(mxval),narg
      DOUBLE PRECISION rval(mxval),secs1,secs2
*
      CHARACTER*10 menu(nkeys)
      CHARACTER*60 help(nkeys)
      DATA menu /'SETESE>   ','ESE*      ','SETEPR>   ','EPR*      ',
     .           'SETPROCES>','PROC*     ',
     .           'DATA>     ','SETPLOT>  ','PLOT*     ',
     .           'HELP      ','EXIT      '/
      DATA HELP
     ./'                                                            ',
     . ' syntax: page                                               ',
     . '                                                            ',
     . ' syntax: page                                               ',
     . '                                                            ',
     . ' syntax: frompage topage                                    ',
     . '                                                            ',
     . ' (not yet implemented)                                      ',
     . ' syntax: frompage topage                                    ',
     . ' Produces this text                                         ',
     . ' (Exit)                                                     '/

      DATA ipos/1/
*********************************************************************
*
*========>>>>>  start user monitor loop
*
      ircode = 0
    1 continue
      ok = .true.
      if(maclevel.le.1) call message(ircode)
      CALL macin(nwords,maclevel)
      IF(maclevel.le.1) THEN
         call recog(ipos,menu,nkeys,index,noccur)
         if(noccur.ne.1) goto 1313
      ELSE
         call mgetcode(index)
         if(merror.ne.0) then
           !write(6,*) ' MACRO EXECUTION ABORTED, error-code=', merror
           maclevel = 0
           goto 1
         endif
      ENDIF

*===>>> COMMAND DISPATCHER
      goto(10,20,30,40,50,60,70,80,90,100,110)index
      WRITE(6,*) ' illegal index in MAINMENU ', index
      stop
*
*===========>>>>>>>>>> execute commands <<<<<<<<<<<<<<==========
*
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> SETESE * ESE program simulation parameters
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   10 continue
*     !write(6,*) ' SETSIM ',maclevel
      if(maclevel.eq.1) call mputcode(index)
      call setsim(nwords,ipos+1,maclevel,ircode)
      goto 9999
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> ESE* ipage     SIMULATE ESE spectrum on page
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   20 continue
      call macparse(nwords,ipos+1,maclevel,index,1,0,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,1)
      if(maclevel.eq.2) then
         !write(6,*) ' test simul page:',ival(1)
      else if(maclevel.ne.1) then
         call clock(secs1)
         call specsim(ival(1))
         call clock(secs2)
         if(maclevel.ne.3) then
           !write(6,'(A,F6.2)') ' EXECUTION TIME:',secs2-secs1
           !write(6,*) ' SUMMARY OF ACCUMULATION PROCESS:'
           call cumstat(6)
         endif
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> SETEPR * EPR program simulation parameters
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   30 continue
      if(maclevel.eq.1) call mputcode(index)
      call setepr(nwords,ipos+1,maclevel,ircode)
      goto 9999
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> EPR* ipage     SIMULATE EPR spectrum on page
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   40 continue
      call macparse(nwords,ipos+1,maclevel,index,1,0,ival,rval,
     .              vindexI,vindexR,narg)
      if(narg.le.0) goto 1313
      call mderefI(ival,vindexI,1)
      if(maclevel.eq.2) then
         !write(6,*) ' test eprsim page:',ival(1)
      else if(maclevel.ne.1) then
         call clock(secs1)
         call epr1sim(ival(1))
         call clock(secs2)
         if(maclevel.ne.3) then
           !write(6,'(A,F6.2)') ' EXECUTION TIME:',secs2-secs1
         endif
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> SETPROC nskip,ppmode,damp0,damp1
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   50 continue
      call macparse(nwords,ipos+1,maclevel,index,2,2,ival,rval,
     .              vindexI,vindexR,narg)
      IF(narg.lt.0) goto 1313
      IF(narg.gt.0) THEN
         call mderefR(rval,vindexR,2)
         call mderefI(ival,vindexI,2)
         call setpp(rval(1),rval(2),ival(1),ival(2))
      ELSE
         call listpp(6)
         goto 1515
      ENDIF
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> PROCESS ipag1,ipag2
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   60 continue
      call macparse(nwords,ipos+1,maclevel,index,2,0,ival,rval,
     .              vindexI,vindexR,narg)
      IF(narg.le.0) goto 1313
      call mderefI(ival,vindexI,2)
      if(maclevel.eq.2) then
         !write(6,*) ' test postproc ',ival(1),ival(2)
      else
         call specproc(ival(1),ival(2))
      endif
      goto 4711
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> DATA  goto data menu
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   70 continue
*     !write(6,*) ' DATA MENU',maclevel
      if(maclevel.eq.1) call mputcode(index)
      call datamen(nwords,ipos+1,maclevel,ircode)
      goto 9999
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> SETPLOT goto setplot menu  (not macro active)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   80 continue
      !write(6,*) ' SETPLOT (not yet implemented)',maclevel
*     call setplot(nwords,ipos+1,maclevel,ircode)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*---> PLOT  frompage topage (not macro active)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   90 continue
      if(nwords-ipos.lt.2) goto 1313
      call getiray(ipos+1,ival,vindexI,2,ok)
      if(.not.ok) goto 1313
      call mderefI(ival,vindexI,2)
      call dataplot(ival(1),ival(2))
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> HELP
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  100 continue
      call helplst(menu,help,nkeys)
*     call comlist(menu,nkeys)
      goto 1515
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*----> EXIT
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  110 continue
      if(maclevel.eq.1) then
         call mputcode(index)
      else
         return
      endif
      goto 4711
*
 4711 continue
      ircode = 0
      goto 9999
 1515 ircode = 0
      if(maclevel.eq.1) ircode = 1
      goto 9999
 1313 continue
      ircode = -1
 9999 continue
      if(maclevel.eq.1.and.ircode.ne.0) call rmcode
      goto 1
      END
