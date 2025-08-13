/* grid3d.c
   [cartesian_grid] = grid3d(kx,ky,kz,d,N1,N2,N3,gridOS,kernelWidth,beta);
   Matlab mex file for performing gridding.
   Inputs:
      kx - DOUBLE VECTOR of kx values, same length as ky and kz
      ky - DOUBLE VECTOR of ky values, same length as kx and kz
      kz - DOUBLE VECTOR of kz values, same length as kx and ky
      d - DOUBLE VECTOR of complex data samples
      N1  - INTEGER grid-size in x dimension
      N2  - INTEGER grid-size in y dimension
      N3  - INTEGER grid-size in z dimension
      gridOS - INTEGER grid over-sampling factor (power of 2)
      kernelWidth - DOUBLE width of Kaiser-Bessel kernel
      beta = DOUBLE Kaiser-Bessel parameter
Typical numbers:
beta = 18.5547;
gridOS = 2;
kernelWidth=4;
 */

#include "mex.h"
#include "math.h"
#include "matrix.h"
#define min(x1,x2) ((x1) > (x2))? (x2):(x1)
#define max(x1,x2) ((x1) > (x2))? (x1):(x2)

/*
>From Numerical Recipes in C, 2nd Edition
*/
double bessi0(double x)
{
    double ax,ans;
    double y;
    
    if ((ax=fabs(x)) < 3.75)
    {
        y=x/3.75,y=y*y;
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
    }
    else
    {
        y=3.75/ax;
        ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1+y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1+y*0.392377e-2))))))));
    }
    return ans;
}

void grid3d(const int NK, const double kx[], const double ky[], const double kz[], const double dR[], const double dI[], const int N1, const int N2, const int N3, const int gridOS, const double kernelWidth, const double beta, double cartesian_gridR[], double cartesian_gridI[])
{
    int indexK, index1, index2, index3;
    double shiftedKx, shiftedKy, shiftedKz, distX, kbX, distY, kbY, kbXY, distZ, kbZ, kbXYZ;
    
    for (indexK = 0; indexK < NK; indexK++)
    {
        shiftedKx = ((double)gridOS)*(kx[indexK]+((double)N1)/2);
        shiftedKy = ((double)gridOS)*(ky[indexK]+((double)N2)/2);
        shiftedKz = ((double)gridOS)*(kz[indexK]+((double)N3)/2);
        for (index1 = (int)(max(0,ceil(shiftedKx - kernelWidth*((double)gridOS)/2))); (index1 <= (int)(min(gridOS*N1-1,floor(shiftedKx + kernelWidth*((double)gridOS)/2))))&&(index1>=0); index1++)
        {
            distX = fabs(shiftedKx - ((double)index1))/((double)gridOS);            
            kbX = bessi0(beta*sqrt(1.0-(2.0*distX/kernelWidth)*(2.0*distX/kernelWidth)))/kernelWidth;
            if (isnan(kbX))
                kbX=0;
            for (index2 = (int)(max(0,ceil(shiftedKy - kernelWidth*((double)gridOS)/2))); (index2 <= (int)(min(gridOS*N2-1,floor(shiftedKy + kernelWidth*((double)gridOS)/2))))&&(index2>=0); index2++)
            {
                distY = fabs(shiftedKy - ((double)index2))/((double)gridOS);
                kbY = bessi0(beta*sqrt(1.0-(2.0*distY/kernelWidth)*(2.0*distY/kernelWidth)))/kernelWidth;
                if (isnan(kbY))
                    kbY=0;
                kbXY = kbX*kbY;
                for (index3 = (int)(max(0,ceil(shiftedKz - kernelWidth*((double)gridOS)/2))); (index3 <= (int)(min(gridOS*N3-1,floor(shiftedKz + kernelWidth*((double)gridOS)/2))))&&(index3>=0); index3++)
                {
                    distZ = fabs(shiftedKz - ((double)index3))/((double)gridOS);
                    kbZ = bessi0(beta*sqrt(1.0-(2.0*distZ/kernelWidth)*(2.0*distZ/kernelWidth)))/kernelWidth;
                    if (isnan(kbZ))
                        kbZ=0;
                    kbXYZ = kbXY*kbZ;
                   
                    cartesian_gridR[index1 + gridOS*N1*index2 + gridOS*gridOS*N1*N2*index3] += kbXYZ*dR[indexK];
                    cartesian_gridI[index1 + gridOS*N1*index2 + gridOS*gridOS*N1*N2*index3] += kbXYZ*dI[indexK];                 
                } 
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
    double *kx, *ky, *kz, *dR, *dI, kernelWidth, beta, *cartesian_gridR, *cartesian_gridI, *temp;
    int NK, N1, N2, N3, gridOS;
 
    NK = mxGetM(prhs[0]);
    kx = mxGetPr(prhs[0]);
    ky = mxGetPr(prhs[1]);
    kz = mxGetPr(prhs[2]);
    
    dR = mxGetPr(prhs[3]);
    if (!(dI = mxGetPi(prhs[3])))
    {
        dI=mxCalloc(NK, sizeof(double));
    }
    
    temp = mxGetPr(prhs[4]);
    N1 = (int)round(*temp);

    temp = mxGetPr(prhs[5]);
    N2 = (int)round(*temp);
    
    temp = mxGetPr(prhs[6]);
    N3 = (int)round(*temp);    
    
    temp = mxGetPr(prhs[7]);
    gridOS =(int)round(*temp);
    
    temp = mxGetPr(prhs[8]);
    kernelWidth = *temp;
        
    temp = mxGetPr(prhs[9]);
    beta = *temp;
    
    mexPrintf("NK=%d, N1=%d, N2=%d, N3=%d, gridOS=%d, kernelWidth=%f, beta=%f\n",NK,N1,N2,N3,gridOS,kernelWidth,beta);
    
    plhs[0] = mxCreateDoubleMatrix(N1*N2*N3*gridOS*gridOS*gridOS,1,mxCOMPLEX);
    cartesian_gridR = mxGetPr(plhs[0]);
    cartesian_gridI = mxGetPi(plhs[0]);
    
    grid3d(NK, kx, ky, kz, dR, dI, N1, N2, N3, gridOS, kernelWidth, beta, cartesian_gridR, cartesian_gridI);
    
    mexPrintf("Returning\n");
    return;
}
