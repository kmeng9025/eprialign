	function [P_opt,P_opt_err] = cw_grad_min(funct, pars)
% function [Popt,P_opt_err] = grad_min(funct,Pf,float,[X Y],{P_extra})
% Pf is varied to produce P_opt that optimizes the fit of Y_hat to Y
% this uses marquardt's method combined with a gradient search
% Hartley's modification is also available but commented out
% this function uses  Y_hat = funct(X,Pf,{P_extra})
% {P_extra} is as many seaprate arguments as you want, just extend the list on call
% Pf is the full parameter list 
% NEVER set an initial value in Pf to zero, give "reasonable" numbers to give sensitivity to the variable
% float determines which elements in Pf get floated and which are fixed
% float=1 means to float that parameter,  =0 means a fixed parameter
% Y is the target (experimental) data that is modeled by Y_hat
% sigma^2 = (Y-s*Y_hat-b)'*(Y - s*Y_hat-b) /N is minimized
% where s, and b are the best scale and offset factors
% ADDED BY CM 
% P_opt_err = errors in fitted values of P_opt cm 8/24/99
% ALL variables are column vectors.
% modofied 2/22/2001 to change eval to feval for Y_not and Z1 to allow compiler to work
% modified P_opt_err calc sj 9/15/2004 
% USE:  requires f_pack.m  and 'funct.m'
%keyboard

P_opt = pars.PG(pars.idx);

Y = pars.yy -mean(pars.yy);
n = length(P_opt);
fctn = 0.01;
tolerance = 1E-6;
criterion = 1.;
lamda = 1.e-3;
new_p = 1;

% new_p is true = 1; new_p=0 is false
% develop the epsilon matrix for taking the derivatives.
eps = fctn*P_opt;
%if an element in P is zero, set eps=fctn
indx = find(P_opt==0);
if(~isempty(indx)), eps(indx) = fctn; end
eps = diag(eps);

% generate initial guesses for the scale factor and baseline
% Y_hat = scale*Y_not + Base
%keyboard
% eval( [ 'Y_not =  ' funct fun_arg ]); 2/22/2001 replace with feval form
%keyboard
Y_not=funct(P_opt,pars);
Z = Y_not -  mean(Y_not);
scale = (Z'*Y)/(Z'*Z);
Y=Y/scale; scale=1;
base = mean(Y) - scale*mean(Y_not);
errs = (Y-scale*Y_not-base);
variance = errs'*errs;

% criterion
% tolerance

while ( criterion > tolerance )
  if(new_p)
    % 'loop',  count=count+1, criterion, tolerance;

    % evaluate Y_hat at a given P vector.
    % Develop the Z matrix = d(Y_hat)/dP
    %keyboard

    Z = [];
    %P_opt
    %n
    for k=1:n
      P1 = P_opt + eps(:,k);
      Z1 = funct(P1,pars);
      Z1 = ( Z1 - Y_not) / eps(k,k);
      Z = [ Z Z1];
    end
    % append effects of scale and base on Z and Y_not:
    Z = [ scale*Z  Y_not  ones(size(Y_not)) ];
    %'built Z'
    %keyboard
    alpha = Z'*Z ;
    beta = Z'*errs;

  end

  % Marquardt's method is to multiply the factor to the diagonal elements of Z
  %keyboard

  del_P = ( alpha + lamda*diag(diag(alpha)) ) \ beta;
  %keyboard
  Y_not = funct(P_opt+del_P(1:n),pars);
  errs = Y -(scale+del_P(n+1))*Y_not - (base+del_P(n+2));
  new_var = errs'*errs;
  criterion = abs(variance - new_var )/variance;
  
  % keyboard
  if new_var <= variance
    lamda = lamda/10;
    P_opt = P_opt + del_P(1:n); new_p = 1;
    scale = scale + del_P(n+1); base = base + del_P(n+2);
    criterion = (variance - new_var )/variance;
    variance = new_var;
  else lamda = 10*lamda; new_p = 0;
  end

  % the Hartley method to find a scale factor (cf) on del_P
  %to guarantee that cf*del_P gives a better minimum than del_P
  %	   Pf = f_pack((P_opt+del_P/2),P_fix,float);
  %	   eval( [ 'Z1 =  ' funct fun_arg ]);
  %	   errs = (Y-Z1);, var(2) = errs'*errs;
  %	   Pf = f_pack((P_opt+del_P),P_fix,float);
  %	   eval( [ 'Z1 =  ' funct fun_arg ]);
  %	   errs = (Y-Z1);, var(3) = errs'*errs;
  %	variance = var(1);
  %	var = var-var(1);
  %	   cf = cfm * var(2:3);
  %	   cf = cf(1)/cf(2);
  %	P_opt = P_opt + cf*del_P;
  % keyboard

end
% P_opt = f_pack(P_opt,P_fix,float);

if(nargout == 2)

  %  We've minimized chisq with the vector P_opt. Perturbing
  %  the N fitted parameters gives the del_chisq function,
  %  del_chisq = chisq(perturbed P_opt) - chisq(P_opt),
  %  which is distributed as a chi-square function with
  %  N degrees of freedom.  Choose a confidence limit, here
  %  68.3%; then find delta = chi2inv(.683,N) so that the probability of a chi-square
  %  variable with N degrees of freedom being less than delta is 68.3%.
  %  Now in the equation: delta = dP_opt'*alpha*dP_opt (eq. I), where alpha
  %  is one-half times the Hessian matrix (which we approximate by
  %  the 'derivative matrix', 'alpha' = Z'*Z above, multiplied by one over
  %  the point variances sig^2; note the alpha of this routine is not the alpha
  %  of eq. I), the vectors dP_opt
  %  define a contour of constant del_chisq in a N dimensional parameter space,
  %  and the projections of these vectors onto the desired parameter axes
  %  give the 'error bars' for those parameters (see Numerical Recipes, section
  %  section 15.6, 'Confidence Limits on Estimated Model Parameters').  This projection of the
  %  ellipse onto the i'th parameter axis corresponds to the i'th diagonal
  %  element of the covariance matrix, which is the inverse of alpha.  This
  %  I think, although standard, is not obvious except from the name
  %  'covariance matrix', and not explained in Num. Rec.  The way to see this
  %  is:
  %  Project the ellipse onto the i'th axis.  This means setting
  %  d(delta)/dP_opt(k) == 0  (eq. II) for all k not equal (.ne.) to i; solving these
  %  N-1 linear equations for P_opt(k.ne.i) in terms of P_opt(k); and then
  %  plugging these into the equation for delta and solving for P_opt(k).
  %  There is a way to do this by partitioning the matrix alpha:
  %
  %   alpha  =  | a  b^t |
  %             | b  d   |
  %
  %  where a is 1x1, b is (N-1)x1, and d is (N-1)x(N-1); a corresponds
  %  to the desired axis of projection.  Partition dP_opt in the equation for
  %  delta:
  %
  %  dP_opt = | x |
  %           | y |
  %
  %  where x is 1x1 and corresponds to the projection axis, and y is
  %  (N-1)x1.  Now Eq. II ==>
  %
  %  | b  d | = | x | = 0   ==>
  %             | y |
  %
  %  y = -x d^-1 b
  %
  %  Plugging this into (I):
  %
  %  delta = | x  y | * | a  b^t | * | x |     ==>
  %                     | b  d   |   | y |
  %
  %  x^2 = delta  *  (a  -  b^t * d^-1 * b)^-1
  %
  %  so that x is the projection of the ellipse onto the i'th axis, and thus
  %  the confidence interval for P_opt(i).
  %  In fact, x corresponds to the i'th diagonal element of the inverse of
  %  the matrix alpha (the covariance matrix).  This can be seen by using
  %  the formulas for matrix 'inversion by partitioning' in NR, section 2.7,
  %  'Sparse Linear Systems'.
  %  However, alpha, as defined in NR and in this comment, is not available here (in this
  %  subroutine); we need to multiply the derivative matrix (called alpha in
  %  this subroutine, but not equal to the alpha of NR) by 1/variance
  %  (variance is the point variance, sig^2), and then use the prescription
  %  above.  Equivalently, we can multiply the P_errs returned by this
  %  subroutine by some factor in the caller function.  But it gets tricky because
  %  of the scale factor.  So alpha(NR) = alpha(here)/(sig^2/scale^2).  Since
  %  the convariance matrix C = alpha(NR)^-1 = sig^2/scale^2 * alpha(here)^-1,
  %  P_errs(actual) = sqrt(diag(C)) = sig/scale*P_errs(here).  Entonces, we
  %  must multiply the P_errs which grad_min returns by sig/scale in the
  %  caller program (I guess you could pass this info in, but this way seems
  %  easier now).  By tbe by, sig = xtra_info(5) in cw-speak, the base line
  %  noise.  These are the changes made in one_opt, many_opt, and
  %  many_opt_image (which was compiled, so that the .dll version is called), and
  %  grad_min_image.
  %  I put the results of a monte carlo for this stuff in the Colin_cw
  %  directory on imaging1: P_errs_statistics.xls.
  %  SJ 7/14/04

  % stuff below added 8/24/99 by cm to include errors
  % generate the linearized errors in P_opt
  %	std_err = sqrt( variance/length(Y) );
  %   P_err = std_err*sqrt(diag(inv(alpha))) ;
  len=length(pars.idx);
  delta = chi2inv(.683,len);
%   P_opt_err = sqrt(delta)*sqrt(diag(inv(alpha)));
  
  res = funct(P_opt, pars);
  Amp = (pars.yy'*res)/(res'*res);
  STD_norm = sqrt(sum((pars.yy/Amp - res).^2)/(length(res)-len));
  
  P_opt_err = STD_norm*sqrt(delta)*sqrt(diag(inv(alpha)));

  % n=size of P_opt = number of variables floated
  % size of P_err = n+2
  % pick first n elements of P_err as the errors
  % is this OK? I think so
  P_opt_err = P_opt_err(1:n);
end