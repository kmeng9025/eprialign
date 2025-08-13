
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mex.h"

static float ****allocate_4d(int ni, int nj, int nk, int nl)
{
   float ****theMatrix;
   int i,j,k,l;

   theMatrix=(float****)malloc(ni*sizeof(float***))-1;
   for (i =  1; i <= ni; i++) {
    theMatrix[i] = (float***) malloc(nj*sizeof(float**))-1 ;
    for (j = 1; j <= nj; j++) {
      theMatrix[i][j]=(float**)malloc(nk*sizeof(float*))-1;
      for (k=1; k<= nk; k++) {
        theMatrix[i][j][k]=(float *)malloc(nl*sizeof(float))-1 ;
	  }
   }
  }

  return theMatrix;
}

void deallocate_4d(float ****theMatrix, int ni, int nj, int nk, int nl)
{
  int i,j,k,l;

   for (i =  1; i <= ni; i++) {
    for (j = 1; j <= nj; j++) {
      for (k=1; k<= nk; k++) free(theMatrix[i][j][k]+1);
	  free(theMatrix[i][j]+1);
	}
	free(theMatrix[i]+1);
  }
  free(theMatrix+1);
  return;
}
static float ***allocate_3d(int ni, int nj, int nk)
{
   float ***theMatrix;
   int i,j,k;

   theMatrix=(float***)malloc(ni*sizeof(float**))-1;
   for (i =  1; i <= ni; i++) {
    theMatrix[i] = (float**) malloc(nj*sizeof(float*))-1 ;
    for (j = 1; j <= nj; j++) {
      theMatrix[i][j]=(float*)malloc(nk*sizeof(float))-1;     
	  }
   }
  

  return theMatrix;
}
static int ***allocate_3d_int(int ni, int nj, int nk)
{
   int ***theMatrix;
   int i,j,k;

   theMatrix=(int***)malloc(ni*sizeof(int**))-1;
   for (i =  1; i <= ni; i++) {
    theMatrix[i] = (int**) malloc(nj*sizeof(int*))-1 ;
    for (j = 1; j <= nj; j++) {
      theMatrix[i][j]=(int*)malloc(nk*sizeof(int))-1;     
	  }
   }
  

  return theMatrix;
}

void deallocate_3d(float ***theMatrix, int ni, int nj, int nk)
{
  int i,j,k;

   for (i =  1; i <= ni; i++) {
    for (j = 1; j <= nj; j++)  free(theMatrix[i][j]+1);    
	free(theMatrix[i]+1);
  }
  free(theMatrix+1);
  return;
}

void deallocate_3d_int(int ***theMatrix, int ni, int nj, int nk)
{
  int i,j,k;

   for (i =  1; i <= ni; i++) {
    for (j = 1; j <= nj; j++)  free(theMatrix[i][j]+1);    
	free(theMatrix[i]+1);
  }
  free(theMatrix+1);
  return;
}

static void ComputeIndexSet(int nstb,int nendb,float centerb,int nsts,int nends,
  float centers,float fkhalf, int kdimu, int manang,float *cta,float *sta, 
    float d1, int ****indx, float ****delta)
{
    
  int ib, is, iib,iis,ia, kxi,ind;
  float sca, bsa;
  float xi,b,s,dist;
  
  *indx=allocate_3d_int(nendb-nstb+1, nends-nsts+1, manang);
  *delta=allocate_3d(nendb-nstb+1, nends-nsts+1, manang);
  
     for (ib=nstb,iib=1; ib <= nendb; ib++,iib++) 
	  {
		b=-centerb+1.0*ib;
		for (is=nsts,iis=1; is <=nends; is++,iis++) 
		{
		  s=-centers+1.0*is;      
		  for (ia=1; ia<=manang; ia++) 
		  {
			sca=s*cta[ia];
			bsa=b*sta[ia];
			xi=sca+bsa;
            (*indx)[iis][iib][ia] = 0;
            if (fabs(xi) < fkhalf)
            {
				xi=xi+fkhalf;   
				kxi=floor(xi);
				if(kxi< 1) continue; /*kxi=1;*/
                if (kxi > kdimu-1) continue; /*kxi=kdimu-1; */
				dist=xi-kxi; 
				ind=kxi+(ia-1)*d1;
                (*indx)[iis][iib][ia] = ind;
                (*delta)[iis][iib][ia] = dist;
            }
		  } /* loop over ia */
		 } /* loop over is */
	  }  /* loop over ib */
   return;
}
/*

C     A MATLAB MEX routine for 4d serial backprojection */
      
	  


	void bp_xds_c_single(float *sino, float nplr, float naz, float nspec,
	   float pri, float ai, float si, float nbins, float bc, float imt, float *image)
	{

	int mtnang,mpnang,manang,kmd,imtype,nmd,ndimu,kdimu;
	
	float theta0,phi0,alpha0,center,fkhalf,fnhalf,dtheta,dphi,dalpha,s,b,x,y,xi;
	float sca, bsa, sct, zst, xcf, ysf;
    float *btemp2;
	float ****btemp1; 
	float ***delta;
    int ***indx;
	char string[20];
	float *ctt, *stt, *ctf, *stf, *cta, *sta,*bfilt,*tfilt;
    extern float ****allocate_4d();

	
	int nxsize, nysize, nzsize, nbsize;
	int nstx, nendx, nsty, nendy, nstz, nendz, nstb, nendb;
	int nbfilt, ntfilt;
	int i, it, ip, ib, is, iib, ia, indexv, ndim;
    int j,k,l,nn;
	int ix, iy, iix,iiy;
	int kxi, iz, iiz, nxi;
    int d1,d2,d3;
    int p1, p2, p3;
    int q1,q2,q3;
    int i0;
    
	float pi = 3.141592653589793238L, theta, phi, alpha;
	float dist, term,  z;
	float t1, t2, t3;
    float sum;
    char msgbuf[256];
    
    int nlhs;
    mxArray *rhs;
    
	mtnang=floor(nplr);
	mpnang=floor(naz);
	manang=floor(nspec);
	kmd=floor(nbins);
	imtype=floor(imt);
	center=bc;
	nmd=kmd;
	ndimu=nmd;
	kdimu=kmd;

    sprintf(msgbuf,"%f %f %f %f %f %f %f %f %f\n",  nplr,  naz, 
         nspec, pri,  ai,  si,  nbins,  bc,  imt);
    rhs=mxCreateString(msgbuf);
    mexCallMATLAB(0,NULL,1,&rhs,"disp");
    
    /* allocate arrays for the sines, cosines and filters.  note
       we want to use fortran-style one-based indexing, so subtract
       one from each pointer. 
    */
    
	ctt = (float *) malloc(mtnang*sizeof(float)) -1;
	stt = (float *) malloc(mtnang*sizeof(float)) -1;
	ctf = (float *) malloc(mpnang*sizeof(float)) -1;
	stf = (float *) malloc(mpnang*sizeof(float)) -1;
	cta = (float *) malloc(manang*sizeof(float)) -1;
	sta = (float *) malloc(manang*sizeof(float)) -1;
	bfilt=(float *) malloc(manang*sizeof(float)) -1;
    tfilt = (float *) malloc(mtnang*sizeof(float)) -1;
	
/*In our data acquisition the integral including the origin is nbins/2
   fkhalf=float(kdimu)/2.
*/

	fkhalf=center;
	fnhalf=1.0*(ndimu+1)/2.0;
	  
/*	Initialize array sizes*/
	nxsize= nysize= nzsize=1;
	nbsize=ndimu;
/*	Start and end planes for B, which is spectral*/
      nstb=1;
      nendb=ndimu;
	switch(imtype) {
	  case 7: 
			nstx = nendx = nsty = nendy = fnhalf;
			nstz = 1;
			nendz = nzsize = ndimu;
			nbfilt= ntfilt=0;
			break;
	  case 6:
			nstx = nendx = fnhalf;
			nsty = 1;
			nendy = ndimu;
			nstz = nendz = fnhalf;
			nysize = ndimu;
			nbfilt= ntfilt=0;
			break;
	  case 5:
			nstx = 1;
			nendx = ndimu;
			nsty = nendy = nstz = nendz = fnhalf;
			nxsize = ndimu;
			nbfilt= ntfilt=0;
			break;
	  case 4:
			nstx = 1;
			nendx = ndimu;
			nsty = 1;
			nendy = ndimu;
			nstz = nendz = fnhalf;
			nxsize = nysize = ndimu;
			nbfilt=1;
			ntfilt=0;
			break;
	  case 3:
			nstx = nendx = fnhalf;
			nsty = 1;
			nendy = ndimu;
			nstz = 1;
			nendz = nysize = nzsize = ndimu;
			nbfilt=1;
			ntfilt=0;
			break;
	  case 2:
			nstx = 1;
			nendx = ndimu;
			nsty = nendy = fnhalf;
			nstz = 1;
			nendz = nxsize = nzsize = ndimu;
			nbfilt=1;
			ntfilt=0;
			break;
	  case 1:
			nstx = nsty = nstz = 1;
			nendx = nendy = nendz = ndimu;
			nxsize = nysize = nzsize = ndimu;
			nbfilt=2;
			ntfilt=1;
			break;
	  case 14:
			nstb = nendb = fnhalf;
			nstx = nsty = nstz = 1;
			nendx = nendy = nendz = ndimu;
			nxsize = nysize = nzsize = ndimu;
			nbsize = 1;
			nbfilt=0;
			ntfilt=1;
			break;
	}  /*end of switch */


/*	Set up the initial angles based on the image type*/
	dtheta=pi/(1.0*mtnang);
 /*   theta0=(pi-dtheta)/2.0;*/
    theta0=pi/2- pri;
	dphi=pi/(1.0*mpnang);
/*    phi0=(pi-dphi)/2.0;*/
	phi0=pi/2- ai;
	dalpha=pi/(1.0*manang);
 /*   alpha0=(pi-dalpha)/2.0;*/
	alpha0=pi/2- si;


	switch(imtype) { 
	  case 7:
			theta0 = phi0 = 0.0;
			ndim = 2;
			break;
	  case 6:
			theta0 =phi0 = pi/2;
			ndim = 2;
			break;
	  case 5:
			theta0 = pi/2;
			phi0 = 0.0;
			ndim = 2;
			break;
	  case 4:
			theta0 = pi/2;
			phi0 = phi0;
			ndim = 3;
			break;
	  case 3:
			theta0 = theta0;
			phi0 = pi/2;
			ndim = 3;
			break;
	  case 2:
			theta0 = theta0;
			phi0 = 0.0;
			ndim = 3;
			break;
	  case 1:
			theta0 = theta0;
			phi0 = phi0;
			ndim = 4;
			break;
	  case 14:
			alpha0 = pi/2;
			theta0 = theta0;
			phi0 = phi0;
			ndim = 3;
			break;

    }  /* end of switch */

/*	Defines the theta angular interval and the initial angle. It saves all 
		the sin and cos values. The initial angle is 90-.5dtheta.
		theta is incremented by -dtheta */
		
      for (i=1; i <=mtnang; i++) {
	   theta=theta0-(i-1)*dtheta;
	   theta=pi/2-theta;
       ctt[i]=cos(theta);
       stt[i]=sin(theta);
       tfilt[i]=pow(fabs(ctt[i]),1.0*ntfilt); /* filter */
	}
    
/*	Defines the phi anglular interval and the initial angle. It saves all 
		the sin and cos values. The initial angle is 90-.5dphi.
		theta is incremented by -dphi*/

     for (i=1; i<= mpnang; i++) {
      phi=phi0-(i-1)*dphi;
      ctf[i]=cos(phi);
      stf[i]=sin(phi);
    }

/*	Defines the alpha anglular interval and the initial angle. It saves all 
		the sin and cos values. The initial angle is 90-.5dalpha.
		alpha is incremented by -dalpha*/
		
      for (i=1; i<=manang; i++) {
        alpha=alpha0-(i-1)*dalpha;
	    alpha=pi/2-alpha;
        cta[i]=cos(alpha);
        sta[i]=sin(alpha);
        bfilt[i]=pow(fabs(cta[i]),1.0*nbfilt);  /* filter */
	  }

    btemp1=allocate_4d(mtnang,mpnang,nbsize,ndimu);
    /*btemp1=(float *) malloc(mtnang*mpnang*nbsize*ndimu*sizeof(float));*/
	t1=1.0*clock()/CLOCKS_PER_SEC;
	mexPrintf("Entering first loop\n");

/*	Below, the origin for i* indices is a cube corner
		the origin for * indices is the cubic center */
    
    d1=kmd; d2=kmd*manang; d3=kmd*manang*mpnang;

    ComputeIndexSet(nstb,nendb,fnhalf,1,ndimu,fnhalf,fkhalf,kdimu,manang,cta,sta,d1,&indx,&delta);
    
    p1=mtnang; p2=mpnang*p1; p3=p2*nbsize;
    
	for (it=1; it <=mtnang; it++)
	 for (ip=1; ip <=mpnang; ip++)
     {
       i0=(ip-1)*d2+(it-1)*d3-1;
      for (ib=nstb,iib=1; ib <= nendb; ib++,iib++) 
	  {
	    /* iib=ib-nstb+1;
		b=-fnhalf+1.0*ib;*/
		for (is=1; is <=ndimu; is++) 
		{
		  /*s=-fnhalf+1.0*is;*/
		  /*btemp1[it][ip][iib][is]=0.0 ;*/
          sum=0.0;
		  for (ia=1; ia<=manang; ia++) 
		  {
            if (indx[is][iib][ia] < 1) continue;
            
/* this is the original code now replaced by ComputeIndexSet outside the loop

            sca=s*cta[ia];
			bsa=b*sta[ia];
			xi=sca+bsa;
			if(fabs(xi) < fkhalf)   
			{
				xi=xi+fkhalf;   
				kxi=floor(xi);
				if(kxi< 1) kxi=1;
                if (kxi > kdimu-1) kxi=kdimu-1; 
				indexv=kxi+(ia-1)*d1+(ip-1)*d2+(it-1)*d3-1;    */
            
                indexv=indx[is][iib][ia]+i0;
                dist=delta[is][iib][ia];
				term=sino[indexv]*(1.0-dist)+sino[indexv+1]*dist; 

                term=term*bfilt[ia];  /* filter if needed */
                sum+=term;


              /*} end of "if (kxy < 1)..."*/
		  } /* loop over ia */
          btemp1[it][ip][iib][is]=sum/manang;
		 } /* loop over is */
	  }  /* loop over ib */
     } /*loop over ip*/

    deallocate_3d_int(indx,nendb-nstb+1,ndimu,manang);
    deallocate_3d(delta, nendb-nstb+1,ndimu,manang);
 
    
	t2=1.0*clock()/CLOCKS_PER_SEC;    
	mexPrintf(" second loop, clock=%f\n", t2-t1);
    btemp2=(float *)malloc(mpnang*ndimu*nzsize*nbsize*sizeof(float));
/*	Below, the origin for i* indices is a cube corner
		the origin for * indices is the cubic center*/
    ComputeIndexSet(nstz,nendz,fnhalf,1,ndimu,fnhalf,fnhalf,ndimu,mtnang,ctt,stt,0,&indx,&delta);
    q1=mpnang; q2=ndimu*q1; q3=q2*nzsize;
	for (ip=1; ip <=mpnang; ip++)
	 for( ib=1; ib <=nbsize; ib++)
      for ( iz=nstz,iiz=1; iz <=nendz; iz++,iiz++)
	  {
		for (is=1; is <=ndimu; is++)
		{
         sum=0.0;
		 for (it=1; it <= mtnang; it++)
		 {
             nxi=indx[is][iiz][it];
             if (nxi < 1) continue;
           
              dist=delta[is][iiz][it];
			  term=btemp1[it][ip][ib][nxi]*(1.0-dist)+btemp1[it][ip][ib][nxi+1]*dist; 
			  term=term*tfilt[it];
              sum += term;
		 }  /* for it */
         *(btemp2+ip + (is-1)*q1 + (iiz-1)*q2 + (ib-1)*q3 -1) = sum/mtnang;
	   }  /* for is */
	 }  /* for iz */
    deallocate_3d_int(indx,nendz-nstz+1,ndimu,mtnang);
    deallocate_3d(delta, nendz-nstz+1,ndimu,mtnang);

	deallocate_4d(btemp1,mtnang,mpnang,nbsize,ndimu);

    t3=1.0*clock()/CLOCKS_PER_SEC;
	mexPrintf(" third loop, clock= %f\n", t3-t1);

/*	Below, the origin for i* indices is a cube corner
		the origin for * indices is the cubic center*/
    d1=nxsize; d2=nxsize*nysize; d3=nxsize*nysize*nzsize;
    ComputeIndexSet(nstx,nendx,fnhalf,nsty,nendy,fnhalf,fnhalf,ndimu,mpnang,stf,ctf,0,&indx,&delta);
	for(ib=1; ib <=nbsize; ib++)
	for (iz=1; iz <=nzsize; iz++)
    {
       i0=(iz-1)*q2 + (ib-1)*q3 -1;
        for (ix=nstx,iix=1; ix <=nendx; ix++,iix++)
        {	
          for (iy=nsty,iiy=1; iy <= nendy; iy++,iiy++)
          {		    
            indexv=iix+(iiy-1)*d1+(iz-1)*d2+(ib-1)*d3-1;	
             sum=0.0;
             for (ip=1; ip <= mpnang; ip++)
             {
                  nxi=indx[iiy][iix][ip];
                  if (nxi < 1) continue;			  
                  dist=delta[iiy][iix][ip];

                  term=*(btemp2 + i0 + ip + (nxi-1)*q1 ) * (1.0-dist) +
                     *(btemp2 + i0+ip + nxi*q1) * dist;			 

                  sum += term;
            }  /* for (ip */
             *(image+indexv)=sum/mpnang;
          }  /* for (iy */
        }  /* for(ix */
    } /* for iz */
    deallocate_3d_int(indx,nendx-nstx+1,nendy-nsty+1,ndimu);
    deallocate_3d(delta,nendx-nstx+1,nendy-nsty+1,ndimu);
  
     t3=1.0*clock()/CLOCKS_PER_SEC;
	mexPrintf(" DONE. clock=%f\n", t3-t1);
  /*deallocate_4d(btemp2,mpnang,ndimu,nzsize,nbsize);*/
    free(btemp2);
	
	return;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

      
      float *image_pr;
	  int ndims;
      int dims_image[4];
      int i, m, n, size, image_size;
      float *sino_pr;
	  float nplr, naz, nspec, nbins, imtype, plri, azii, speci, bc;

/*C     Check for proper number of arguments. */

      if (nrhs != 10) 
          mexErrMsgTxt("10 inputs required. \
		   [sino,np,na,ns, plri,azii,speci,nb,bin_center,type]");
	  if (nlhs != 1) 
		mexErrMsgTxt("One output required. [image]");
      

/*     Check to insure the input arrays are numeric (not strings).*/
      if (!mxIsNumeric(prhs[0])) mexErrMsgTxt("Input 1 must be a numeric array.");
      if (!mxIsNumeric(prhs[1])) mexErrMsgTxt("Input 2 must be a numeric array.");
      if (!mxIsNumeric(prhs[2])) mexErrMsgTxt("Input 3 must be a numeric array.");
      if (!mxIsNumeric(prhs[3])) mexErrMsgTxt("Input 4 must be a numeric array.");
      if (!mxIsNumeric(prhs[4])) mexErrMsgTxt("Input 5 must be a numeric array.");
      if (!mxIsNumeric(prhs[5])) mexErrMsgTxt("Input 6 must be a numeric array.");
      if (!mxIsNumeric(prhs[6])) mexErrMsgTxt("Input 7 must be a numeric array.");
	  if (!mxIsNumeric(prhs[7])) mexErrMsgTxt("Input 8 must be a numeric array.");
	  if (!mxIsNumeric(prhs[8])) mexErrMsgTxt("Input 9 must be a numeric array.");
	  if (!mxIsNumeric(prhs[9])) mexErrMsgTxt("Input 10 must be a numeric array.");


/*	Get the sinogram dimensions*/
	nplr = mxGetScalar(prhs[1]);
	naz = mxGetScalar(prhs[2]);
	nspec = mxGetScalar(prhs[3]);
	nbins = mxGetScalar(prhs[7]);
	imtype = mxGetScalar(prhs[9]);

/*	Get the initial angles*/
	plri = mxGetScalar(prhs[4]);
	azii = mxGetScalar(prhs[5]);
	speci = mxGetScalar(prhs[6]);
	bc = mxGetScalar(prhs[8]);


/*     Get the size of the sinogram.*/
      m = mxGetM(prhs[0]);
      n = mxGetN(prhs[0]);
      size = m*n;
#if 0
mexPrintf("m,n,nplr,az,spec,bins= %d %d %f %f %f %f\n",
      m,n,nplr,naz,nspec,nbins);
#endif

/*     Sinogram must be of the given dimensions*/
    if (size != nplr*naz*nspec*nbins) mexErrMsgTxt("Input dimensions are inconsistent.");

      
/*     Create matrix for the return argument.*/

	sino_pr = (float *)mxGetPr(prhs[0]);
	ndims=floor((13.0-imtype)/3.0);
	if (imtype == 14) ndims=3;

	image_size=1;
	for (i=1; i <=ndims; i++) 
    {
        dims_image[i-1]=nbins;
        image_size*=nbins;
    }
/*
        dims_image[0] = image_size;
        dims_image[1] = 1;
 */
    
/*		mexPrintf("%d %d\n", dims_image[0], dims_image[1]);*/

        plhs[0] = mxCreateNumericArray(ndims,dims_image,mxSINGLE_CLASS,mxREAL);
       image_pr = (float *)mxGetPr(plhs[0]);

/*     Call the computational subroutine.*/

	  bp_xds_c_single(sino_pr,nplr,naz,nspec,plri,azii,speci,nbins,
	      bc,imtype,image_pr);


      return;
}

