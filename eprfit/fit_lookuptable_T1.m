% function [fitted] = fit_lookuptable_T1(fit_y,tt,TT)
% 'fit_y' is MxN matrix with data from voxels to be fit (M = # voxels,
%    N = # data points for each voxel to be fit
% 'tt' are 2*taus used
% 'TT' are T delay values

function [fitted] = fit_lookuptable_T1(fit_y,tt,TT)

%**********************************************************************
%---------------------------Make Dictionary-------------------------
%**********************************************************************

% delta_pO2 = delta_R*(1000*1.84)/(2*pi*2.8)
numT = length(TT);
T1tau = tt(1)/2;
R1 = 0.001:0.0001:1.61;  %R2=1.61 gives pO2=149.6
T1 = 1./R1';
tempT1 = repmat(T1,1,numT);
tempT = repmat(TT,size(tempT1,1),1);
T1dict = 1-2*exp(-tempT./tempT1);
clear tempT1 tempT
tempnormcoeff = repmat(max(T1dict,[],2),1,numT); %Normalize inversion recoveries to maximum point (simulate full recovery)
T1dict = T1dict./tempnormcoeff;
tempnormcoeff = repmat(1-min(T1dict,[],2),1,numT); %Normalize (1-inversion recoveries) (simulate full flip)
T1dict = 1 - T1dict;
T1dict = T1dict./tempnormcoeff;
tempnormcoeff = repmat(sqrt(sum(T1dict.^2,2)),1,numT); %Normalize final library vectors
T1dict = T1dict./tempnormcoeff;
clear tempnormcoeff

%***************************************************[*******************
%-------------------Use Dictionary : Find T2/T1------------------------
%**********************************************************************
%Find T1
decaydata = fit_y;
normcoeff = max(decaydata,[],2); %Normalize inversion recoveries to maximum point (simulate full recovery)
normcoeff = repmat(normcoeff,1,numT);
decaydata = decaydata./normcoeff;
normcoeff = repmat(1-min(decaydata,[],2),1,numT); %Normalize (1-inversion recoveries) (simulate full flip)
fitted.inv = normcoeff(:,1); %B in A(1-Bexp(-TT/T1))
decaydata = 1 - decaydata;
decaydata = decaydata./normcoeff;
lookup = decaydata*T1dict';
% lookup finds dot product for data from each voxel with each possible decay curve in dictionary
% each row for one voxel, each column lookups to entry in dictionary
clear normcoeff decaydata
[~,maxidx] = max(lookup,[],2); % Finds the index for entry in parameter list with maximum dot product
fitted.T1 = T1(maxidx); % Look up corresponding T1 that gave curve with max dot product
%Find Amp (not yet working)
fitted.amp = max(fit_y,[],2); %A in A(1-Bexp(-TT/T1))
T2effect = exp(-2*T1tau/fitted.T1)'; %Find approximate T2 influence assuming T2 = T1
fitted.amp = fitted.amp./T2effect;

%**********************************************************************
%-------------Match output data to standard output data----------------
%**********************************************************************
% inversion previously calculated is B in S = A(1-B*exp(-TT/T1))
% inversion found in standard fitting is B' in S = A' - B'*exp(-TT/T1),
% (A'=A, B' = AB)
fitted.inv = fitted.inv.*fitted.amp;
% switch dimensions to match standard fitting output:
fitted.amp = fitted.amp';
fitted.T1 = fitted.T1';
fitted.inv = fitted.inv';

%**********************************************************************
%-------------------Calculate Residuals/Error--------------------------
%**********************************************************************
% degree_of_freedom = length(TT) - 2;
% % exponential = X - Y*exp(-TT/T1)
% % d(exponential)/d(T1)
% fit_exp_dT1 = repmat(fitted.inv',1,length(TT)).*...
%     exp(-repmat(TT,size(fit_y,1),1)./repmat(fitted.T1',1,length(TT))).*...
%     repmat(TT,size(fit_y,1),1)./repmat(fitted.T1',1,length(TT))./...
%     repmat(fitted.T1',1,length(TT));
% %d(exponential)/d(Y)
% fit_exp_dY = exp(-repmat(TT,size(fit_y,1),1)./repmat(fitted.T1',1,length(TT)));
% %d(exponential)/d(X)
% fit_exp_dX = ones(size(fit_exp_dY));
% fitresult = repmat(fitted.amp',1,length(TT)) - ...
%     repmat(fitted.inv',1,length(TT)).*exp(-repmat(TT,size(fit_y,1),1)./...
%     repmat(fitted.T1',1,length(TT)));
% rmse = sqrt(sum((fit_y - fitresult).^2,2));
% residual_std = rmse/sqrt(degree_of_freedom);
% fitted.error = zeros(3,size(fit_y,1));
% for ii = 1:size(fit_y,1)
%     J = [fit_exp_dX(ii,:); fit_exp_dT1(ii,:); fit_exp_dY(ii,:)]';
%     sigma = residual_std(ii)^2*inv(J'*J);
%     se = sqrt(diag(sigma))';
%     fitted.error(:,ii) = se;
% end

