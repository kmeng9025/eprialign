// ConvGrid.cpp : Mex function generator cooresponding to ConvGrid.m
// 05-06-2013 by Hyungseok Jang (jang35@wisc.edu)

/* $Revision: 1.10.6.6 $ */
#include <math.h>
#include "mex.h"

/* Input Arguments */

#define	KDATA		prhs[0]
#define	GRIDX		prhs[1]
#define	GRIDY		prhs[2]
#define	GRIDRES		prhs[3]
#define	DENSCOMP	prhs[4]
#define	KERNEL		prhs[5]
#define	KERNRES		prhs[6]

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

static void LibConvGrid(double kspaceReal[], 
					 double kspaceImag[], 
					 double kdataReal[], 
					 double kdataImag[], 
					 double gridX[], 
					 double gridY[],
					 double gridRes[],
					 double densComp[],
					 double gridKern[],
					 int kernSize,
					 int kernRes,
					 int numPoints
					 )
{
/*=====================================================================
% parameters
%   kdata - the data as an [numPoints,1] vector
%   gridX - the kx coordinates for the data
%   gridY - the ky corrdinates for the data
%   gridRes - the size of the gridded data, NxN
%   densComp - density compensation function from ConvGridDensity
%   gridKern - kernel struct from ConvKernel
%	kernSize - size of kernel
%	kernRes - resolution of kernel
% output
%   kspace - the output kspace data
========================================================================*/

//% To-do : check inputs 
//numPoints = length(gridX);
//if ( length(gridY) ~= numPoints )
//    error('invalid grid inputs, gridX and gridY must be the same size\n');
//end
//if ( size(kdata,1) ~= numPoints )
//    error('invalid kdata input, must have size [%g,1]',numPoints);
//end
//if ( size(kdata,2) ~= 1 )
//    error('invalid kdata input, must have size [%g,1]',numPoints);
//end
//if ( sum( size(densComp) == gridRes ) ~= 2 )
//    error('density compensation function must be of size gridRes X gridRes');
//end

int kernCen = floorx((kernSize+1)/2); 
int gridSize = (int)gridRes[0];
//mexPrintf("%d, %d",kernSize, kernRes);

for( int i=0; i<numPoints; i++ ) {
    double currX = gridX[i];
    double currY = gridY[i];

    // loop through the griddable points on the kernel
    for(int kky=-floorx(kernRes/2); kky<=floorx(kernRes/2); kky++){
        int kLocY = roundx(currY + kky);
        double kernLocY = (kLocY-currY)*kernSize/kernRes + kernCen;        

	    for(int kkx=-floorx(kernRes/2); kkx<=floorx(kernRes/2); kkx++){
            int kLocX = roundx(currX + kkx);
            double kernLocX = (kLocX-currX)*kernSize/kernRes + kernCen;        

			// skip invalid points
	        if ( kLocY < 1 )
	            continue;
			else if ( kLocX < 1 )
				continue;
			else if ( kernLocY < 1 )
				continue;
			else if ( kernLocX < 1 )
				continue;
			else if ( kLocY > gridSize )
				continue;
			else if ( kLocX > gridSize )
				continue;
			else if ( kernLocY > kernSize )
				continue;
			else if ( kernLocX > kernSize )
				continue;
       
			//bi-linear interpolation of the kernel
			int tmpQX = ceilx(kernLocX)-floorx(kernLocX);
			int tmpQY = ceilx(kernLocY)-floorx(kernLocY);
			int tmpQ  = tmpQX * tmpQY;
			double tmpX1 = ceilx(kernLocX)-kernLocX;
			double tmpX2 = kernLocX-floorx(kernLocX);
			double tmpY1 = ceilx(kernLocY)-kernLocY;
			double tmpY2 = kernLocY-floorx(kernLocY);
			double intKernel = 0;

			if ( (tmpQX == tmpQY) && (tmpQ == 0 ) ) // no interpolation needed
				intKernel = gridKern[ roundx(kernLocY-1) + roundx(kernLocX-1) * kernSize ];
			else if ( tmpQX == 0 ) // interp in Y 
				intKernel = tmpY1/tmpQY*gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1) * kernSize ] + 
							tmpY2/tmpQY*gridKern[ ceilx(kernLocY-1) + floorx(kernLocX-1) * kernSize ];
			else if ( tmpQY == 0 ) // interp in X
				intKernel = tmpX1/tmpQX*gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1) * kernSize ] + 
							tmpX2/tmpQX*gridKern[ floorx(kernLocY-1) + ceilx(kernLocX-1) * kernSize ];
			else // interp in X+Y
				intKernel = gridKern[ floorx(kernLocY-1) + floorx(kernLocX-1) * kernSize ] / tmpQ * tmpX1 * tmpY1 + 
							gridKern[ floorx(kernLocY-1) + ceilx(kernLocX-1) * kernSize ] / tmpQ * tmpX2 * tmpY1 + 
							gridKern[ ceilx(kernLocY-1) + floorx(kernLocX-1) * kernSize ] / tmpQ * tmpX1 * tmpY2 + 
							gridKern[ ceilx(kernLocY-1) + ceilx(kernLocX-1) * kernSize ] / tmpQ * tmpX2 * tmpY2;
            
        // convolve
        double currKReal = kdataReal[i] * intKernel * densComp[ (kLocY-1) + (kLocX-1) * gridSize ];
        double currKImag = kdataImag[i] * intKernel * densComp[ (kLocY-1) + (kLocX-1) * gridSize ];

		kspaceReal[ (kLocY-1) + (kLocX-1) * gridSize ] += currKReal;
		kspaceImag[ (kLocY-1) + (kLocX-1) * gridSize ] += currKImag;
		}
	}
}
return;

}


void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *kspaceReal, *kspaceImag; 
    double *kdataReal, *kdataImag, *gridX, *gridY, *gridRes, *densComp, *kernel, *kernRes;
    size_t m, n, Nout, kernSize; 
    
    /* Check for proper number of arguments */
    
    if (nrhs != 7) { 
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
    KSPACE = mxCreateDoubleMatrix( (mwSize)Nout, (mwSize)Nout, mxCOMPLEX);     

    /* Assign pointers to the various parameters */ 
    kspaceReal = mxGetPr(KSPACE);
    kspaceImag = mxGetPi(KSPACE);
	kdataReal = mxGetPr(KDATA);
	kdataImag = mxGetPi(KDATA);
    gridX = mxGetPr(GRIDX);
    gridY = mxGetPr(GRIDY);
    gridRes = mxGetPr(GRIDRES);
    densComp = mxGetPr(DENSCOMP);
    kernel = mxGetPr(KERNEL);
    kernRes = mxGetPr(KERNRES);

    /* Do the actual computations in a subroutine */
    LibConvGrid(kspaceReal,kspaceImag, kdataReal, kdataImag, gridX, gridY, 
		gridRes, densComp, kernel, kernSize, kernRes[0], m*n); 
    return;
}

