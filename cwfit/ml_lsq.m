      function [ Z , coef , s, Cons ] = ml_lsq( Data , power )
%  [ Z, coef, s , {Confd_intvl} ] = ml_lsq ( Data , Power );
% ml_lsq does a generalized multi linear least squares fit to Data = [ X Y ]
% Data is a matrix of matched data [X Y1(X) Y2(X)  ...]. (two or many columns)
% It is made from the column vectors X and Y(X) by typing: Data = [X Y]
% The fit to Y ( Y_fit) is in the matrix Z = [ X Y Y_fit ], on output.
% If Power is a  scalar: power is the order of the polynomial in the fitting.
% If Power is a ROW vector: the list of exponents to be used in the fitting
% Power is  (x^power) for each element in B  eg. B(0), the Y intercept.
% coef == [ power , B , err_B ] is a matrix created by the call. 
% It contains the coefficients B of the polynomial best fit (the 2nd column), 
% The values "err_B" are the standard deviations of each of the coefficients.
% These errors are corrected for degrees of freedom, phi == N - P. (see Choice)
% sigma is the rms deviation between Y and Yfit (aka the standard error)
% sigma_est is the best estimator to sigma corrected for the degrees of freedom
% R is the multiple correlation coefficient NOT corrected for the mean.
% Rm is the multiple correlation coefficient corrected for the mean
% s == [ sigma sigma_est R  Rm ]'
% Confd_intvl are the x and y confindence intervals as [ Conf_x  Conf_y ]
% they are optional if desired.  Need to multiply by student's t.

% generate the matrix: M, M is the Model Matrix
%  coef = inverse(M'*M) * (M'*Y)
% the model is Y_fit = M * coef
%  choice == % choose whether you want the standard error (choice=1)
% or the best estimator to the standard error (choice=2) when generating 
% confidence invervals for both the coefficients and the x and y variables

        choice = 2;, [ N , n ] = size(Data);
        X = Data(:,1);, Y = Data(:,2:n);, %  N == the number of data points
        if( length(power) > 1), Pow = power;, else, Pow = [ 0:power ];, end
        lp = length(Pow);,  phi = N-lp;  % phi == the degrees of freedom
        M =  [];
        for rs = 1:lp,  M = [ M  ( X .^ Pow(rs)) ];,   end
        A = M' * M;, coef = A \ ( M' * Y );,  
        var = inv(A); % the variance covarinace matrix
        Z = M * coef; % this is Y_hat the fit to the Ys.
        eps_cw = Y - Z; % the individual errors at all the data points
        %eps is a built in function.  changed to eps_cw by CH 1-20-06
        s =  diag(eps_cw'*eps_cw);
        s =   [ 1/N ; 1/phi ] * s';
        R =  diag(Z'*Y) ./  diag(Y'*Y) ;
        Rm = diag( corrcoef([Z,Y]),n-1) .^ 2;
        s = [ s ; [R Rm ]' ];
        svc = s(choice,:);

if(nargout == 4)
        MP = [];
        for rs = 1:lp, MP = [MP ( Pow(rs) * ( X .^ (Pow(rs)-1) ) )];, end
        sy_hat =   diag( (M * var * M') ) ;
        Yprime = MP * coef;
        sx_hat = ( ( 1 + sy_hat ) * svc ) ./ ( Yprime .^ 2 );
        Cons = sqrt ( [ sx_hat  (sy_hat*svc) ] ) ;
% need to multiply Cons == Confidence intervals for both x and y
%  by student's t == student_t(alpha, phi)  to get the full 
%  confidence intervals.   The y ones are for the normal use
%  the x column of Cons is for the inverse regression problem.
end
        Z = [ Data Z ];
        coef = [ Pow' coef  sqrt( diag( var ) * svc ) ];
        s = sqrt(s);
%  keyboard

