#include "mex.h"

/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  double *Qnumbers, *Rule, *Out,tmp,tmp1;
  int     Qrows, Qcols, Rrows, Rcols;
  int     i,j,k,l,match,nTrans = 0,idx;
  char *errmsg;
  
  /* Debug*/
  /*char buffer[200];*/
  
  /*  Check for proper number of arguments. */
  /* NOTE: You do not need an else statement when using
     mexErrMsgTxt within an if statement. It will never
     get to the else statement if mexErrMsgTxt is executed.
     (mexErrMsgTxt breaks you out of the MEX-file.) 
  */
  if(nrhs != 2) 
    mexErrMsgTxt("Two inputs required: x=binning(y, x, binx, biny).");
  if(nlhs != 1) 
    mexErrMsgTxt("One output required.");
  
  /* Get the dimensions of the Quantum number array. */
  Qnumbers = mxGetPr(prhs[0]); 
  Qrows = mxGetM(prhs[0]);
  Qcols = mxGetN(prhs[0]); /* For future. */

  /* Get the dimensions of the Rules array. */
  Rule = mxGetPr(prhs[1]);
  Rrows = mxGetM(prhs[1]);
  Rcols = mxGetN(prhs[1]);
  
  if(Qcols!=Rcols) 
    mexErrMsgTxt("Wrong Rules dimension.");

  /* Prepare output array */
  plhs[0] = mxCreateDoubleMatrix(2,1000, mxREAL);
  Out=mxGetPr(plhs[0]);

  for(i=0; i<Qrows-1; i++)	
   for(j=i+1; j<Qrows; j++)	
     {
	   for(k=0; k<Rrows; k++)
	    {
			match = 1;
			for(l=0; l<Rcols; l++)
			 {
			  tmp = Qnumbers[i+l*Qrows]-Qnumbers[j+l*Qrows];
			  tmp = tmp < 0 ? -tmp : tmp;
			  tmp1 = Rule[k+l*Rrows];
			  tmp1 = tmp1 < 0 ? -tmp1 : tmp1;
			  tmp = tmp - tmp1;
			  tmp = tmp < 0 ? -tmp : tmp;
			  if(tmp > 0.1) {match=0;break;}
			 }  
		   if(match == 1)
			{
/*			  sprintf(buffer, "disp('i=%d j=%d')",i,j);
			  mexEvalString(buffer);*/
			  Out[nTrans*2]=i+1; 
			  Out[nTrans*2+1]=j+1;
			  nTrans++;	
			}
		}
	 }
   mxSetN(plhs[0], nTrans);
 } 
  
