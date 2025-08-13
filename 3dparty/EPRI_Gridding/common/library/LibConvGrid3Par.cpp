// ConvGrid3.cpp : Mex function generator cooresponding to ConvGrid3.m
// 01-07-2014 by Hyungseok Jang (jang35@wisc.edu)
#include "pthread.h" /* for threading */
/* $Revision: 1.10.6.6 $ */
//#include <windows.h>
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
#define NUMWORKER   prhs[8]

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

// argument sutructure used for pthread
struct arg_struct {
    double *densComp;
    double *kspaceReal;
    double *kspaceImag;
    double *kdataReal;
    double *kdataImag;
    double *gridX;
    double *gridY;
    double *gridZ;
    double *gridKern;
    int gridSize;
    int kernSize;
    int kernRes;
    int numPoints;
    int startIndex;
    int endIndex;
    int thIndex;
};



pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
void *convolvePoints(void *arguments)
{
    struct arg_struct *args = (struct arg_struct *)arguments;
    double *densComp = args->densComp;
    double *kdataReal = args->kdataReal;
    double *kdataImag = args->kdataImag;
    double *kspaceReal = args->kspaceReal;
    double *kspaceImag = args->kspaceImag;
    double *gridX = args->gridX;
    double *gridY = args->gridY;
    double *gridZ = args->gridZ;
    double *gridKern = args->gridKern;
    int kernSize = args->kernSize;
    int gridSize = args->gridSize;    
    int kernRes = args->kernRes;
    int numPoints = args->numPoints;
    int startIndex = args->startIndex;
    int endIndex = args->endIndex;
    int thIndex = args->thIndex;
    int kernCen = floorx((kernSize+1)/2);
    
    for( int i=startIndex; i<=endIndex; i++ ) {
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
                    
                    //pthread_mutex_lock(&mutex);
                    kspaceReal[ (kLocY-1) + (kLocX-1) * gridSize + (kLocZ-1) * gridSize*gridSize ] += currKReal;
                    kspaceImag[ (kLocY-1) + (kLocX-1) * gridSize + (kLocZ-1) * gridSize*gridSize ] += currKImag;
                    //pthread_mutex_unlock(&mutex);
                }}}
    }// end of sampling point loop
    
    return 0;
}


static void LibConvGrid3Par(double kspaceReal[],
        double kspaceImag[],
        double kdataReal[],
        double kdataImag[],
        double gridX[],
        double gridY[],
        double gridZ[],
        int gridRes,
        double densComp[],
        double gridKern[],
        int kernSize,
        int kernRes,
        int numPoints,
        int numWorker
        )
{
    if(numPoints < 1)
        return;
    
    // threads for parallelization
    pthread_t *threads = (pthread_t*) malloc(sizeof(pthread_t)*numWorker);
    int chunkSize = roundx(numPoints/numWorker);
    
    // Setup arguments for threads
    struct arg_struct * args = (struct arg_struct*) malloc(sizeof(struct arg_struct)*numWorker);
    
    // use multiple threads
    int rc, t;
    for(t=0; t<numWorker; t++) {
        args[t].densComp = densComp;
        args[t].kdataReal = kdataReal;
        args[t].kdataImag = kdataImag;
        args[t].kspaceReal = kspaceReal;
        args[t].kspaceImag = kspaceImag;
        args[t].gridX = gridX;
        args[t].gridY = gridY;
        args[t].gridZ = gridZ;
        args[t].gridKern = gridKern;
        args[t].gridSize = gridRes;
        args[t].kernSize = kernSize;
        args[t].kernRes = kernRes;
        args[t].numPoints = numPoints;
        args[t].thIndex = t;
        args[t].startIndex = chunkSize*t;
        if(t<numWorker-1)
            args[t].endIndex = chunkSize*(t+1)-1;
        else
            args[t].endIndex = numPoints-1;
        
        //mexPrintf("Creating thread %d\n", t);
        rc = pthread_create(&threads[t], NULL, &convolvePoints, (void*)&args[t]);
        if(rc)
            mexErrMsgTxt("problem with return code from pthread_create()");
        //sleep(1); /* wait some time before making the next thread */
    }
    
    for(t=0; t<numWorker; t++)
        pthread_join(threads[t], NULL);
    
    return;
}


void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
        
{
    double *kspaceReal, *kspaceImag, *kdataReal, *kdataImag;
    double *gridX, *gridY, *gridZ, *gridRes, *densComp, *kernel, *kernRes, *numWorker;
    size_t m, n, Nout, kernSize;
    
    /* Check for proper number of arguments */
    
    if (nrhs != 9) {
        mexErrMsgIdAndTxt( "MATLAB:ConvGrid:invalidNumInputs",
                "9 input arguments required.");
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
    numWorker = mxGetPr(NUMWORKER);
    
    /* Do the actual computations in a subroutine */
    LibConvGrid3Par(kspaceReal,kspaceImag, kdataReal, kdataImag, gridX, gridY, gridZ,
            (int)gridRes[0], densComp, kernel, kernSize, kernRes[0], m*n, (int)numWorker[0]);
    return;
}

