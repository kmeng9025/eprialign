function  xtra_info =  cw_xtra_info(xx,yy,n_parts, xf)
% function  xtra_info =  m_bs_noi(ds,n_parts, xf)
% xf and xtra_info are both xtra_info as used elsewhere
% this routine updates xtra_info
% this routine scans the datasets in ds and
% returns what one might believe is the base line noise
% each data set is broken into n_parts and std_error of each
% is calculated, then the smallest std_error is returned in xtra_info.

[m n] = size(yy);
remainder=rem(m,n_parts);
tempm=m-remainder;
kk = round(tempm/n_parts);
%keyboard
if( tempm ~= kk*n_parts), error('cannot reshape; reset n_parts'), end
xtra_info = xf;
std_err=[ ];
rms_sig = [ ];

for k = 1:n
  %getting the x axis of the current data set
  % use reshape to get the Data wanted:
  y = reshape(yy(1:tempm,k),kk,n_parts);
  %keyboard
  [Z,c,s] =ml_lsq([ xx(1:kk,k) y ],1);
  std_err = [ std_err  min(s(2,:) ) ];
  rms_sig = [ rms_sig std(yy(1:tempm,k)) ];
end

%keyboard
xtra_info(5,:)    = std_err;
% chi squared
xtra_info(6,:) = (xtra_info(4,:) ./ std_err) .^ 2;
% compute the snr as xtra_info(7,mds)
xtra_info(7,: )    =  rms_sig  ./ std_err ;


%  np=get_gl('n_peaks');
%  if np==1
%     xtra_info(9,:)=0;
%  xtra_info(10,:)=0;
%        end
