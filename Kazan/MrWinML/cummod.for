*     CALL cumtest
*     end
*
      SUBROUTINE CUMTEST
      IMPLICIT none
      INTEGER nac,limtab,ntab1,ntab2
      LOGICAL x2
      PARAMETER (ntab1=20,ntab2=6,nac=512)
      DOUBLE PRECISION a1tab(0:ntab1-1),f1tab(0:ntab1-1),fmax,thini,
     .                 a2tab(0:ntab2-1),f2tab(0:ntab2-1),thlim,
     .                 accu(nac)
*
      DATA f1tab /0.0D0,1.1D0,2.0D0,3.1D0,5.0D0,6.1D0,7.2D0,8.1D0,9.1D0,
     .2.1D0,0.1D0,0.5D0,3.3D0,1.6D0,4.6D0,7.1D0,8.0D0,9.9D0,2.2D0,3.4D0/
      DATA a1tab /0.80,0.02,0.005,0.02,0.005,0.02,0.005,0.01,0.005,0.01,
     .            0.01,0.01,0.005,0.01,0.005,0.02,0.005,0.01,0.005,0.02/
      DATA f2tab /0.0D0,1.6D0,5.7D0,4.1D0,6.2D0,9.3D0/
      DATA a2tab /0.5D0,0.1D0,0.1D0,0.1D0,0.1D0,0.1D0/
      x2 = .false.
      fmax = 10.D0
      thini= 0.004D0
      thlim= 0.005D0
      limtab = 50
      write(6,*) 'start...'
      CALL inicum(nac,fmax)
      CALL setcum(thini,thlim,limtab,2,1)
      CALL dzero(accu,nac)
      CALL nuccum(a1tab,f1tab,ntab1,1,x2,1,3,accu,nac,1.D0)
      CALL nuccum(a1tab,f1tab,ntab1,1,x2,2,3,accu,nac,1.D0)
      CALL nuccum(a2tab,f2tab,ntab2,1,x2,3,3,accu,nac,1.D0)
      CALL cumstat(6)
      END

***********************************************************************
*
*     CUMMOD (accumulation and convolution of ESEEM signals)
*
*     INICUM: initialize module
*     SETCUM: set parameters for spectral synthesis (tholds, buflen ..)
*     NUCCUM: convolute ESEEM subspectrum (peak table)
*             and add to accumulator
*     GETCUM: pass accumulator to plotting or printing routine
*             (pbuf,nc,ibuf)    pbuf: Real*4 (plot,fft... etc)
*             >>>> eventueel GETCUMR voor R*4 en GETCUMD voor R*8 <<<<
***********************************************************************

**- - - - - - - - - - - - - - -
*      BLOCK DATA
**- - - - - - - - - - - - - - -
*      IMPLICIT none
*      DOUBLE PRECISION thini,thlim,th
*      INTEGER limtab,idebug,method
*      COMMON /STCUM/ thini,thlim,th,limtab,method,idebug
*      END
*----------------------------------------------------------------------
      SUBROUTINE inicum(nc,fmax)
*----------------------------------------------------------------------
*     Initialize cum-module
*
*     nc:     length of accumulator buffer
*     fmax:   maximum frequency value
*-----------------------------------------------------------------------
      IMPLICIT none
*****v
      INTEGER mxt,mxac,itab,limtab,nac,ntab,iprofile,ntry,
     .              ncalls,naccums,method,idebug
      DOUBLE PRECISION atab,thini,thlim,th,fnyq,acamp,acdc
      PARAMETER (mxt=1024,mxac=1024)
      COMMON /DCUM/ atab(0:mxt,2),fnyq
      COMMON /STCUM/ thini,thlim,th,limtab,method,idebug
      COMMON /ICUM/ itab(0:mxt,2),ntab(2),nac
      COMMON /STAT/ acamp(2),acdc(2),iprofile(10),ntry(3),
     .              ncalls,naccums
*****^
*-->  arguments
      INTEGER nc,limtabx,idebugx,methodx,junit,i
      DOUBLE PRECISION fmax,thinix,thlimx
      if(nc.gt.mxac) stop ' inicum: accumulator dimension > 1024'
      nac = nc
      fnyq = fmax
      call izero(itab,mxt*2+2)
      call dzero(atab,mxt*2+2)
      call izero(iprofile,10)
      call izero(ntab,2)
      call izero(ntry,3)
      call dzero(acamp,2)
      call dzero(acdc,2)
      ncalls = 0
      naccums = 0
      thini=0.005 
      thlim=0.04 
      th=0.005 
      limtab=50
      idebug=0
      method=1
      th = thini
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY setcum(thinix,thlimx,limtabx,methodx,idebugx)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*     thinix: initial threshold (% of max intensity) below which peaks
*             are rejected for further processing
*     thlimx: limiting threshold for activating "emergency procedure"
*     limtabx: limiting number of peaks for activating compress routine
*     method: method of accumulation: 1=table  2=accu  3=fft method
*- - -
      thini  = thinix
      th     = thinix
      thlim  = thlimx
      limtab = limtabx
      idebug = idebugx
      method = methodx
      if(method.lt.0.or.method.gt.3) method = 1
*
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY getcum(thinix,thlimx,limtabx,methodx,idebugx)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      thinix  = thini
      thlimx  = thlim
      limtabx = limtab
      idebugx = idebug
      methodx = method
*
      RETURN
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ENTRY cumstat(junit)
*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*
*     Print statisitics of accumulation and convolution.
*
*     Intensity profile (Histogram): number of orientations falling
*                       in intensity ranges 60-70-........150 %
*     This should be an indication of the intensity lost due to
*     applying a threshold.
*
      write(junit,*) ' number of calls :' , ncalls
      write(junit,*) ' number of accums:' , naccums
      write(junit,*) ' combi retries   :' , ntry(1)
      write(junit,*) ' compress retries:' , ntry(2)
      write(junit,*) ' emergencies     :' , ntry(3)
      write(junit,*) ' threshold       :' , th
      if(naccums.le.0) return
      write(junit,*) ' average amp (a) :' , acamp(1)/naccums
      write(junit,*) ' average amp (b) :' , acamp(2)/naccums
      write(junit,*) ' average dc  (a) :' , acdc (1)/naccums
      write(junit,*) ' average dc  (b) :' , acdc (2)/naccums
      write(junit,*) ' intensity profile 60,70,..100..150%  :'
      write(junit,'(10(1X,I6))') (iprofile(i), i = 1 , 10)
      write(junit,*) ' lost intensity =',
     .   1.D0 -(acamp(1)+acamp(2)+acdc(1)+acdc(2))/naccums
      END

*----------------------------------------------------------------------
      SUBROUTINE nuccum(axtab,fxtab,nxtab,ibuf,x2,inuc,nnuc,accu,nc,
     .                  factor)
*----------------------------------------------------------------------
*     Pass peak (sub)table to cum-mudule
*
*     axtab:    amplitudes
*     fxtab:    frequencies
*     nxtab:    number of peaks (excluding zero)
*     ibuf:     number of accumulation buffer/table
*     factor:   weighing factor for orientation selection
*     x2:       (logical) .true. if two equivalent nuclei
*     inuc/nnuc: call for inuc of total nnuc calls
*              (if last nucleus no further compressing is necessary
*               peak table is added to the accumulation buffer)
*----------------------------------------------------------------------
      IMPLICIT none
*****v
      INTEGER mxt,mxac,itab,limtab,nac,ntab,iprofile,ntry,
     .              ncalls,naccums,method,idebug
      DOUBLE PRECISION atab,thini,thlim,th,fnyq,acamp,acdc
      PARAMETER (mxt=1024,mxac=1024)
      COMMON /DCUM/ atab(0:mxt,2),fnyq
      COMMON /STCUM/ thini,thlim,th,limtab,method,idebug
      COMMON /ICUM/ itab(0:mxt,2),ntab(2),nac
      COMMON /STAT/ acamp(2),acdc(2),iprofile(10),ntry(3),
     .              ncalls,naccums
*****^
*-->  arguments
      LOGICAL x2
      INTEGER ibuf,nxtab,nc,inuc,nnuc
      DOUBLE PRECISION axtab(0:*),fxtab(0:*),accu(0:*),factor
*-->  local
      DOUBLE PRECISION wbuf(0:mxac),awtab(0:mxt),dsum,sum
      INTEGER iwtab(0:mxt),nwtab,isum,i,itry(3)
*
*-->  check parameters
      IF(nc.ne.nac.or.limtab.le.7)
     .   stop ' cummod call with inconsistent NAC and/or LIMTAB'
      IF(ibuf.ne.1.and.ibuf.ne.2)
     .   stop ' incorrect ibuf in call to nuccum'
      IF(nxtab.gt.mxt) stop ' nuccum: nxtab too large'
      IF(factor.lt.1D-4) return
*
      ncalls = ncalls+1
*
*-->  INIT
      call izero(itry,3)
      if(inuc.eq.1) then
         ntab(ibuf) = 0
         th = thini
      endif
*
*-->  transfer table to workarrays
      nwtab   = nxtab
      DO 10 i = 0 , nwtab-1
      awtab(i) = axtab(i)
  10  iwtab(i) = idnint(fxtab(i)*nac/fnyq)
*
      IF(.not.x2) goto 11
      IF(method.eq.1) then
        call combi2(awtab,iwtab,nwtab,mxt,limtab,th,thlim,wbuf,nac,itry)
      ELSE
        call convo2(awtab,iwtab,nwtab,mxt,th,wbuf,nac)
      ENDIF
      if(idebug.eq.1)write(6,*)' nwtab,wsum:',nwtab,dsum(awtab,nwtab)
*
  11  CONTINUE
      IF(inuc.eq.nnuc) THEN
         naccums = naccums + 1
         IF(method.eq.1) THEN
           call combi(atab(0,ibuf),itab(0,ibuf),ntab(ibuf),
     .        mxt,mxt,awtab,iwtab,nwtab,th,thlim,wbuf,nac,itry)
         ELSE
           call convo(atab(0,ibuf),itab(0,ibuf),ntab(ibuf),
     .        mxt,awtab,iwtab,nwtab,th,wbuf,nac)
         ENDIF
         call tab2cum(atab(0,ibuf),itab(0,ibuf),ntab(ibuf),
     .                accu,nac,factor)
      if(idebug.eq.1) write(6,*) ' current ntab:', ntab(ibuf)
**
*-->  update statistics
         acdc(ibuf) = acdc(ibuf) + atab(0,ibuf)
         sum = dsum(atab(1,ibuf),ntab(ibuf)-1)
      if(idebug.eq.1) write(6,*) ' current sum:', sum + atab(0,ibuf)
         acamp(ibuf) = acamp(ibuf) + sum
         isum = idnint((sum+atab(0,ibuf))*10.D0) - 5.0
         if(isum.gt.10) isum = 10
         if(isum.lt.1) isum = 1
         iprofile(isum) = iprofile(isum) + 1
      ELSE
         IF(method.eq.1) THEN
           call combi(atab(0,ibuf),itab(0,ibuf),ntab(ibuf),
     .        mxt,limtab,awtab,iwtab,nwtab,th,thlim,wbuf,nac,itry)
         ELSE
           call convo(atab(0,ibuf),itab(0,ibuf),ntab(ibuf),
     .        mxt,awtab,iwtab,nwtab,th,wbuf,nac)
         ENDIF
      ENDIF
*
      ntry(1) = ntry(1) + itry(1)
      ntry(2) = ntry(2) + itry(2)
      ntry(3) = ntry(3) + itry(3)
      END
*
*----------------------------------------------------------------------
      SUBROUTINE combi2
     .             (atab,itab,ntab,mxtab,limtab,th,thlim,accu,nac,itry)
*----------------------------------------------------------------------
*---> convolute a peak table on it self (ESEEM of two equivalent nuclei)
*     atab  : amplitudes in peak table
*     itab  : frequency index in peak table
*     ntab  : current number of entries (index of next available entry)
*     mxtab : maximum number of entries in table
*     limtab: maxlength of table at exit
*     th    : threshold for combination amplitude to be added (%of max)
*     accu  : accumulator
*     itry(1) : number of retries to fit peaks into atab
*     itry(2) : number of retries to compress table
*     itry(3) : number of emergency calls
*
*----------------------------------------------------------------------
      IMPLICIT none
*---arguments
      DOUBLE PRECISION atab(0:*),dc,th,thlim,alevel,xamp,accu(0:*),
     .                 ampmax
      INTEGER itab(0:*),ntab,mxtab,limtab,nac,itry(3)
*---local
      INTEGER i,j,k,n,ntry
      DOUBLE PRECISION dm
*
*     if(ntab.gt.limtab) stop ' -combi- illegal value for ntab/limtab'
      if(ntab.gt.limtab) goto 1300
      if(ntab.gt.mxtab) stop ' -combi2- illegal value for ntab'
*-->  if table empty (or only contains dc component) no action
      if(ntab.le.1) return
*--> initialize
      ampmax = 0.D0
      ntry = 0
*
*----> find maximum amplitude
      do 10 i = 1 , ntab-1
   10 ampmax = dmax1(ampmax,DABS(atab(i)))
*
*----> ENTRY for retry
*
   11 CONTINUE
      alevel = ampmax*th
*
      dc = 0.D0
      dm = 0.D0
*
*---> generate combination frequencies:
      n = ntab
      DO 21 j = 2 , ntab-1
      DO 20 i = 1 , j-1
      IF (n+1.le.mxtab-1) THEN
            xamp = atab(i)*atab(j)
*     write(6,*) 'i, xamp, th, alevel', i,xamp,th,alevel
          if(DABS(xamp).lt.alevel) goto 20
            atab(n) = xamp
            itab(n) = itab(i) + itab(j)
            n = n + 1
            atab(n) = xamp
            itab(n) = iabs(itab(i)-itab(j))
            n = n + 1
            dm = dm + xamp*2.D0
      ELSE
*           if(ntry.gt.10) goto 1313
            if(th*2.D0.gt.thlim) goto 1300
*       ---> pull up threshold and try again
            th = th * 2.D0
            ntry = ntry + 1
      write(6,*) ntry, th
            goto 11
      ENDIF
   20 CONTINUE
   21 CONTINUE
*
      DO 30 k = 1 , ntab-1
*--->      intensity of double frequencies
           xamp  = atab(k)**2 * 0.5D0
*--->      dc contribution
           dc = dc + xamp
           dm = dm + xamp
*--->      new intensity of basic frequencies
           atab(k) = atab(k) * atab(0) * 2.0D0
           dm = dm + atab(k)
*--->      check intensity of double frequency
           if(DABS(xamp).lt.alevel) goto 30
*--->      check space in peak table
           IF (n.le.mxtab-1) THEN
             atab(n) = xamp
             itab(n)= 2 * itab(k)
             n = n + 1
           ELSE
*            if(ntry.gt.10) goto 1313
             if(th*2.D0.gt.thlim) goto 1300
*       ---> pull up threshold and try again
             th = th * 2.D0
             ntry = ntry + 1
             goto 11
           ENDIF
   30 CONTINUE
      itry(1) = ntry
      ntab = n
*---->update dc-component
      atab(0) = atab(0)**2 + dc
      dm = dm + atab(0)
*
*===> Compress table: reduce ntab below limtab
*
      if(ntab.le.limtab) return
      call compress(atab,itab,ntab,limtab,th,thlim,accu,nac,itry(2))
*
*---->normal termination (ntab and itry set by routine "compress")
      itry(1) = ntry
      RETURN
*
*---->error condition
*1313 ntab = n
*---> emergency procedure
 1300 CONTINUE
      call convo2(atab,itab,ntab,mxtab,th,accu,nac)
      itry(3) = itry(3)+1
      END

*-------------------------------------------------------------------
      SUBROUTINE combi(atab,itab,ntab,mxtab,limtab,ax,ix,nx,
     .                                        th,thlim,accu,nac,itry)
*-------------------------------------------------------------------
*---->Combination of two peak tables (sum and difference frequencies)
*
*     atab  : amplitudes in (current) peak table
*     itab  : frequency index (current) in peak table
*     ntab  : current number of entries (index of next available entry)
*     mxtab : maximum number of entries in (current) table
*     ax    : amplitudes of new table to be combined with current
*     ix    : frequency indices of new table
*     th    : threshold for combination amplitude to be added
*     accu  : work array
*     itry(1) : number of retries to fit peaks into atab
*     itry(2) : number of retries to compress table
*     itry(3) : number of emergency procedures
*
*--------------------------------------------------------------------
      IMPLICIT none
*--- arguments
      INTEGER nx,ntab,mxtab,itry(3),ix(0:*),itab(0:*),nac,limtab
      DOUBLE PRECISION ax(0:*),atab(0:*),th,thlim,accu(0:*)
*--- local
      INTEGER i,j,n,nsave,ntry
      DOUBLE PRECISION ampmax,alevel,xamp
*
      if(ntab.gt.limtab) goto 1300
      if(ntab.gt.mxtab) stop ' -combi1- illegal value for ntab'
*---> if ax/ix empty: nothing to combine:
      if(nx.le.1) return
*---> if table empty, simply copy ax/ix
      IF(ntab.le.1) THEN
         do 10 i = 0 , nx-1
            atab(i) = ax(i)
            itab(i) = ix(i)
  10     CONTINUE
         ntab    = nx
         return
      ENDIF
*
      n = ntab
      ampmax = 0.D0
*---> add basic frequencies of ax and find maximum
      do 50  i = 1 , nx-1
             if(n.gt.mxtab-1) goto 1300
             itab(n)   = ix(i)
             atab(n)   = ax(i)*atab(0)
             n         = n + 1
             ampmax = dmax1(ampmax,DABS(ax(i)*atab(0)))
  50  CONTINUE
*
*---> check if maximum/minimum of current table is higher
*
      do 51 i = 1 , ntab-1
  51  ampmax = dmax1(ampmax,DABS(atab(i)*ax(0)))
*
      ntry = 0
      nsave = n
  55  CONTINUE
      alevel = th * ampmax
      n = nsave
*
*-->  Add combination frequencies
      do 200 j = 1 , ntab-1
      do 100 i = 1 , nx-1
           xamp = atab(j)*ax(i) / 2.0D0
           if(DABS(xamp).lt.alevel) goto 100
             IF(n+1.gt.mxtab-1) THEN
             IF(th*2.D0.gt.thlim) goto 1300
                th = th*2.D0
                ntry = ntry + 1
                goto 55
             ENDIF
*
             itab(n)   = ix(i) + itab(j)
             atab(n)   = xamp
             n         = n + 1
             itab(n)   = IABS(ix(i) - itab(j))
             atab(n)   = xamp
             n         = n + 1
  100 CONTINUE
  200 CONTINUE
*
*---> update intensities of basic frequencies
      do 300 i = 1 , ntab-1
      atab(i) = atab(i) * ax(0)
  300 CONTINUE
*
*---> update dc term
      atab(0) = atab(0) * ax(0)
*
      ntab = n
      itry(1) = ntry
*===> Compress table: reduce ntab below limtab
*
      if(ntab.le.limtab) return
      call compress(atab,itab,ntab,limtab,th,thlim,accu,nac,itry(2))
*
*---> normal termination (itry, ntab set by subr. "compress")
      RETURN
*
*---> ERROR condition
*1313 itry(1) = ntry
 1300 CONTINUE
      CALL convo(atab,itab,ntab,mxtab,ax,ix,nx,th,accu,nac)
      itry(3) = itry(3)+1
      END

*-------------------------------------------------------------------
      SUBROUTINE tab2cum(atab,itab,ntab,accu,nac,factor)
*-------------------------------------------------------------------
*     add frequency intensity table to accumulator buffer
*     factor: weighing for orientation selection
*     itab : frequency (fourier numbers)
*     atab : intensities
*
*-------------------------------------------------------------------
      IMPLICIT none
      INTEGER itab(0:*),ntab,nac,i,ix,ifouri
      DOUBLE PRECISION atab(0:*),accu(0:*),factor
*
      accu(0) = accu(0) + atab(0)*factor
      DO 10 i = 1 , ntab-1
         ix = ifouri(itab(i),nac)
  10     if(ix.lt.nac) accu(ix) = accu(ix) + atab(i)*factor
      END

      FUNCTION ifouri(ix,nac)
      ix = iabs(ix)
      if(ix.le.nac) goto 2
   1  ix = iabs(2*nac-ix)
      if(ix.gt.nac) goto 1
   2  ifouri = ix
      END

*----------------------------------------------------------------------
      SUBROUTINE compress(atab,itab,ntab,limtab,th,thlim,accu,nac,ntry)
*----------------------------------------------------------------------
*===> Compress table: reduce ntab below limtab
*
      IMPLICIT none
      DOUBLE PRECISION atab(0:*),th,thlim,alevel,accu(0:*),ampmax
      INTEGER itab(0:*),ntab,limtab,ntry,nac,i,n
      if(ntab.le.limtab) return
      call dzero(accu,nac+1)
      call tab2cum(atab,itab,ntab,accu,nac,1.D0)
*----> find maximum amplitude
      ampmax = 0.D0
      do 10 i = 1 , nac
   10 ampmax = dmax1(ampmax,DABS(accu(i)))
      ntry = 0
*
      atab(0) = accu(0)
   11 CONTINUE
      n = 1
      alevel = ampmax * th
      DO 20 i = 1 , nac
         if(DABS(accu(i)).lt.alevel) goto 20
         atab(n) = accu(i)
         itab(n) = i
         n = n+1
         IF(n.gt.limtab.and.th*2.D0.lt.thlim) THEN
            th = th * 2.D0
            ntry = ntry + 1
            goto 11
         ENDIF
   20 CONTINUE
*
      ntab = n
      END
*----------------------------------------------------------------------
      SUBROUTINE convo2(atab,itab,ntab,max,th,accu,nac)
*----------------------------------------------------------------------
*---> convolute a peak table on it self (ESEEM of two equivalent nuclei)
*     atab  : amplitudes in peak table
*     itab  : frequency index in peak table
*     ntab  : current number of entries (index of next available entry)
*     max   : maximum number of entries in table and accumulator
*     th    : threshold for combination amplitude to be added (%of max)
*     accu  : accumulator
*----------------------------------------------------------------------
      IMPLICIT none
*---arguments
      DOUBLE PRECISION atab(0:*),th,accu(0:*)
      INTEGER itab(0:*),ntab,max,nac
*---local
      INTEGER i,j,k,ixx,ifouri
      DOUBLE PRECISION dc,dm,alevel,xamp,ampmax
*
*-->  if table empty (or only contains dc component) no action
      if(ntab.le.1) return
*--> initialize
      ampmax = 0.D0
      call dzero(accu,nac+1)
*
*----> find maximum amplitude
      do 10 i = 1 , ntab-1
   10 ampmax = dmax1(ampmax,DABS(atab(i)))
*
   11 CONTINUE
      alevel = ampmax*th
*
      dc = 0.D0
      dm = 0.D0
*
*---> generate combination frequencies:
      DO 21 j = 2 , ntab-1
      DO 20 i = 1 , j-1
            xamp = atab(i)*atab(j)
          if(DABS(xamp).lt.alevel) goto 20
            ixx = ifouri( itab(i) + itab(j),nac)
            accu(ixx) = accu(ixx) + xamp
            ixx = ifouri( itab(i)-itab(j) , nac)
            accu(ixx) = accu(ixx) + xamp
            dm = dm + xamp*2.D0
   20 CONTINUE
   21 CONTINUE
*
      DO 30 k = 1 , ntab-1
*--->      intensity of double frequencies
           xamp  = atab(k)**2 * 0.5D0
*--->      dc contribution
           dc = dc + xamp
           dm = dm + xamp
*--->      new intensity of basic frequencies
           ixx = ifouri(itab(k),nac)
           accu(ixx) = accu(ixx) + atab(k)*atab(0) * 2.D0
           dm = dm + atab(k)
*--->      check intensity of double frequency
           if(DABS(xamp).lt.alevel) goto 30
           ixx = ifouri(2*itab(k),nac)
           accu(ixx) = accu(ixx) + xamp
   30 CONTINUE

*---->update dc-component
      accu(0) = atab(0)**2 + dc
*---->total intensity (for debugging purposes)
      dm = dm + accu(0)
*
      call cum2tab(accu,nac,alevel,atab,itab,ntab,max)
      END

*-------------------------------------------------------------------
      SUBROUTINE convo(atab,itab,ntab,max,ax,ix,nx,th,accu,nac)
*-------------------------------------------------------------------
*---->Combination of two peak tables (sum and difference frequencies)
*
*     atab  : amplitudes in (current) peak table
*     itab  : frequency index (current) in peak table
*     ntab  : current number of entries (index of next available entry)
*     max : maximum number of entries in (current) table
*     ax    : amplitudes of new table to be combined with current
*     ix    : frequency indices of new table
*     th    : threshold for combination amplitude to be added
*     accu  : work array
*--------------------------------------------------------------------
      IMPLICIT none
*--- arguments
      INTEGER nx,ntab,max,ix(0:*),itab(0:*),nac
      DOUBLE PRECISION ax(0:*),atab(0:*),xamp,th,alevel,accu(0:*),
     .                 ampmax
*--- local
      INTEGER i,j,ixx,ifouri
*
*---> if ax/ix empty: nothing to combine:
      if(nx.le.1) return
*---> if table empty, simply copy ax/ix
      IF(ntab.le.1) THEN
         do 10 i = 0 , nx-1
            atab(i) = ax(i)
            itab(i) = ix(i)
  10     CONTINUE
         ntab    = nx
         return
      ENDIF
*
      call dzero(accu,nac+1)
      ampmax = 0.D0
      do 50  i = 1 , nx-1
         xamp = ax(i)*atab(0)
         ixx  = ifouri(ix(i),nac)
         accu(ixx) = accu(ixx) + xamp
         ampmax = dmax1(ampmax,DABS(xamp))
  50  CONTINUE
*
      do 51 i = 1 , ntab-1
         xamp = atab(i)*ax(0)
         ixx  = ifouri(itab(i),nac)
         accu(ixx) = accu(ixx) + xamp
         ampmax = dmax1(ampmax,DABS(xamp))
  51  CONTINUE
*
  55  CONTINUE
      alevel = th * ampmax
*
*-->  Add combination frequencies
      do 200 j = 1 , ntab-1
      do 100 i = 1 , nx-1
           xamp = atab(j)*ax(i) / 2.0D0
         if(DABS(xamp).lt.alevel) goto 100
           ixx  = ifouri(ix(i)+itab(j),nac)
           accu(ixx) = accu(ixx) + xamp
           ixx  = ifouri(ix(i)-itab(j),nac)
           accu(ixx) = accu(ixx) + xamp
  100 CONTINUE
  200 CONTINUE
*
*---> update dc term
      accu(0) = atab(0) * ax(0)
*
      call cum2tab(accu,nac,alevel,atab,itab,ntab,max)
*
      END

*------------------------------------------------------------------
      SUBROUTINE cum2tab(accu,nac,alevel,atab,itab,ntab,max)
*------------------------------------------------------------------
*    Reverse of "tab2cum": transfer accu contence into table

      IMPLICIT none
      INTEGER nac,itab(0:*),ntab,i,n,max
      DOUBLE PRECISION atab(0:*),accu(0:*),alevel
      atab(0) = accu(0)
      n = 1
      DO 10 i = 1 , nac
        if(accu(i).lt.alevel) goto 10
        atab(n) = accu(i)
        itab(n) = i
        n = n + 1
        if(n.gt.max) stop ' cum2tab: table index out of range !'
   10 CONTINUE
      ntab = n
      END
