// ConvGrid3.cpp : Mex function generator cooresponding to ConvGrid3.m
// 05-13-2013 by Hyungseok Jang (jang35@wisc.edu)

/* $Revision: 1.10.6.6 $ */
#include <math.h>
#include "mex.h"

/* Input Arguments */

#define	KDATA		prhs[0]
#define	GRIDX		prhs[1]
#define	GRIDY		prhs[2]
#define	GRIDZ		prhs[3]
#define	GRIDRES		prhs[4]
#define	DENSCOMP	prhs[5]
#define	KERNEL		prhs[6]
#define	KERNRES		prhs[7]

/* Output Arguments */

#define	KSPACE		plhs[0]

int roundx(double x){
    if(x > 0)
        return (int)(x+0.5);
    else
        return (int)(x-0.5);
}
int floorx(double x){
    return (int)floorf(x);
}
int ceilx(double x){
    return (int)ceilf(x);
}

static void LibConvGrid3(double kspaceReal[],
        double kspaceImag[],
        double kdataReal[],
        double kdataImag[],
        double gridX[],
        double gridY[],
        double gridZ[],        
        double gridRes[],
        double densComp[],
        double gridKern[],
        int kernSize,
        int kernRes,
        int numPoints
        )
{
    /*=====================================================================
     * % parameters
     * %   kdata - the data as an [numPoints,1] vector
     * %   gridX - the kx coordinates for the data
     * %   gridY - the ky corrdinates for the data
     * %   gridRes - the size of the gridded data, NxN
     * %   densComp - density compensation function from ConvGridDensity
     * %   gridKern - kernel struct from ConvKernel
     * %   kernSize - size of kernel
     * %   kernRes - resolution of kernel
     * % output
     * %   kspace - the output kspace data
     * ===================================================================*/
if(numPoints < 1)
    return;


int kernCen = floorx((kernSize+1)/2);
int gridSize = (int)gridRes[0];

for( int i=0; i<numPoints; i++ ) {
    double currX = gridX[i];
    double currY = gridY[i];
    double currZ = gridZ[i];

    // loop through the griddable points on the kernel
    for(int kkz=-floorx(kernRes/2); kkz<=floorx(kernRes/2); kkz++){
        int kLocZ = roundx(currZ + kkz);
        double kernLocZ = (kLocZ-currZ)*kernSize/kernRes + kernCen;        

    for(int kky=-floorx(kernRes/2); kky<=floorx(kernRes/2); kky++){
        int kLocY = roundx(currY + kky);
        double kernLocY = (kLocY-currY)*kernSize/kernRes + kernCen;        

    for(int kkx=-floorx(kernRes/2); kkx<=floorx(kernRes/2); kkx++){
        int kLocX = roundx(currX + kkx);
        double kernLocX = (kLocX-currX)*kernSize/kernRes + kernCen;  

        // skip invalid points
        if ( kLocY < 1 || kLocX < 1 || kLocZ < 1 
             || kernLocY < 1 || kernLocX < 1 || kernLocZ < 1 
             || kLocY > gridSize || kLocX > gridSize || kLocZ > gridSize 
             || kernLocY > kernSize || kernLocX > kernSize || kernLocZ > kernSize )
            continue;

        //bi-linear interpolation of the kernel
        int tmpQX = ceilx(kernLocX)-floorx(kernLocX);
        int tmpQY = ceilx(kernLocY)-floorx(kernLocY);
        int tmpQZ = ceilx(kernLocZ)-floorx(kernLocZ);
        int tmpQ  = tmpQX * tmpQY * tmpQZ;
        double tmpX1 = ceilx(kernLocX)-kernLocX;
        double tmpX2 = kernLocX-floorx(kernLocX);
        double tmpY1 = ceilx(kernLocY)-kernLocY;
        double tmpY2 = kernLocY-floorx(kernLocY);
        double tmpZ1 = ceilx(kernLocZ)-kernLocZ;
        double tmpZ2 = kernLocZ-floorx(kernLocZ);
        double intKernel = 0;

        // Interpolate kernel.
        if ( tmpQX == 0 && tmpQY == 0 && tmpQZ == 0 ) // no interpolation needed
            intKernel = gridKern[ roundx(kernLocY-1) + roundx(kernLocX-1)*kernSize + roundx(kernLocZ-1)*kernSize*kernSize ];
        else if ( tmpQX == 0 && tmpQZ == 0 ) // interp in Y
            intKernel = tmpY1/tmpQY*gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] +
                    tmpY2/tmpQY*gridKern[ ceilx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ];
        else if ( tmpQY == 0 && tmpQZ == 0) // interp in X
            intKernel = tmpX1/tmpQX*gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] +
                    tmpX2/tmpQX*gridKern[ floorx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ];
        else if ( tmpQX == 0 && tmpQY == 0) // interp in Z
            intKernel = tmpZ1/tmpQZ*gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] +
                    tmpZ2/tmpQZ*gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ];
        else if ( tmpQX ==0 ) // interp in Y+Z
            intKernel = gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / (tmpQY * tmpQZ) * tmpY1 * tmpZ1 +
                    gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ] / (tmpQY * tmpQZ) * tmpY1 * tmpZ2 +
                    gridKern[ ceilx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / (tmpQY * tmpQZ) * tmpY2 * tmpZ1 +
                    gridKern[ ceilx(kernLocY-1) + floorx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ] / (tmpQY * tmpQZ) * tmpY2 * tmpZ2;
        else if ( tmpQY ==0 ) // interp in X+Z
            intKernel = gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / (tmpQX * tmpQZ) * tmpX1 * tmpZ1 +
                    gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ] / (tmpQX * tmpQZ) * tmpX1 * tmpZ2 +
                    gridKern[ floorx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / (tmpQX * tmpQZ) * tmpX2 * tmpZ1 +
                    gridKern[ floorx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ] / (tmpQX * tmpQZ) * tmpX2 * tmpZ2;
        else if ( tmpQZ ==0 ) // interp in X+Y
            intKernel = gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / (tmpQY * tmpQX) * tmpY1 * tmpX1 +
                    gridKern[ floorx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / (tmpQY * tmpQX) * tmpY1 * tmpX2 +
                    gridKern[ ceilx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / (tmpQY * tmpQX) * tmpY2 * tmpX1 +
                    gridKern[ ceilx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / (tmpQY * tmpQX) * tmpY2 * tmpX2;
        else // interp in X+Y+Z
            intKernel = gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / tmpQ * tmpY1 * tmpX1 * tmpZ1 +
                    gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ] / tmpQ * tmpY1 * tmpX1 * tmpZ2 +
                    gridKern[ floorx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / tmpQ * tmpY1 * tmpX2 * tmpZ1 +
                    gridKern[ floorx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ] / tmpQ * tmpY1 * tmpX2 * tmpZ2 +
                    gridKern[ ceilx(kernLocY-1) + floorx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / tmpQ * tmpY2 * tmpX1 * tmpZ1 +
                    gridKern[ ceilx(kernLocY-1) + floorx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ] / tmpQ * tmpY2 * tmpX1 * tmpZ2 +
                    gridKern[ ceilx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + floorx(kernLocZ-1)*kernSize*kernSize ] / tmpQ * tmpY2 * tmpX2 * tmpZ1 +
                    gridKern[ ceilx(kernLocY-1) + ceilx(kernLocX-1)*kernSize + ceilx(kernLocZ-1)*kernSize*kernSize ] / tmpQ * tmpY2 * tmpX2 * tmpZ2;

        // convolve
        double currKReal = kdataReal[i] * intKernel * densComp[ (kLocY-1) + (kLocX-1) * gridSize + (kLocZ-1) * gridSize*gridSize];
        double currKImag = kdataImag[i] * intKernel * densComp[ (kLocY-1) + (kLocX-1) * gridSize + (kLocZ-1) * gridSize*gridSize];

        int nsubs = 3, subs[3];
        subs[0] = kLocX;
        subs[1] = kLocY;
        subs[2] = kLocZ;

         kspaceReal[ (kLocY-1) + (kLocX-1) * gridSize + (kLocZ-1) * gridSize*gridSize ] += currKReal;
         kspaceImag[ (kLocY-1) + (kLocX-1) * gridSize + (kLocZ-1) * gridSize*gridSize ] += currKImag;
    }}}
}// end of sampling point loop

return;
    
}


void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
        
{
    double *kspaceReal, *kspaceImag;
    double *kdataReal, *kdataImag, *gridX, *gridY, *gridZ, *gridRes, *densComp, *kernel, *kernRes;
    size_t m, n, Nout, kernSize;
    
    /* Check for proper number of arguments */
    
    if (nrhs != 8) {
        mexErrMsgIdAndTxt( "MATLAB:ConvGrid:invalidNumInputs",
                "7 input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgIdAndTxt( "MATLAB:ConvGrid:maxlhs",
                "Too many output arguments.");
    }
    
    Nout = mxGetM(DENSCOMP);
    m = mxGetM(KDATA);
    n = mxGetN(KDATA);
    kernSize = mxGetM(KERNEL);
    
    
    /* Create a matrix for the return argument */
    //KSPACE = mxCreateDoubleMatrix( (mwSize)Nout, (mwSize)Nout, mxCOMPLEX);
    int ndim = 3, dims[3];
    dims[0] = Nout;
    dims[1] = Nout;
    dims[2] = Nout;
    KSPACE = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX);

    /* Assign pointers to the various parameters */
    kspaceReal = mxGetPr(KSPACE);
    kspaceImag = mxGetPi(KSPACE);
    kdataReal = mxGetPr(KDATA);
    kdataImag = mxGetPi(KDATA);
    gridX = mxGetPr(GRIDX);
    gridY = mxGetPr(GRIDY);
    gridZ = mxGetPr(GRIDZ);    
    gridRes = mxGetPr(GRIDRES);
    densComp = mxGetPr(DENSCOMP);
    kernel = mxGetPr(KERNEL);
    kernRes = mxGetPr(KERNRES);
    
    /* Do the actual computations in a subroutine */
    LibConvGrid3(kspaceReal,kspaceImag, kdataReal, kdataImag, gridX, gridY, gridZ,
            gridRes, densComp, kernel, kernSize, kernRes[0], m*n);
    return;
}

