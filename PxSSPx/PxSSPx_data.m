function res = PxSSPx_data(pars)

the_path = safeget(pars, 'path', '');
fidx =  safeget(pars, 'fidx', '');
the_template = safeget(pars, 'ftemplate', '');
use_bl_correction = safeget(pars, 'bl_correction', 0);
if(use_bl_correction)
    disp('Using base line correction');
end
dispstat('','init')

fname = sprintf(['%s',the_template], the_path, fidx(1));
[ax,~,dsc]=kv_d01read(fname);
dispstat('Loading parameters.');

ScanWidth = kvgetvalue(dsc.aliases_RSwidth);
Freq = kvgetvalue(dsc.aliases_RSfrequency);
sampling = mean(diff(ax.x));

ii=1;
for kk=fidx
  fname = sprintf(the_template, kk);
  dispstat(['Loading: ',fname]);
  [ax,y,dsc]=kv_d01read(fullfile(the_path, fname));
  [x_RS,y_RS(:,ii)]=rs_sdeconvolve(y, ScanWidth, Freq, sampling, pars);
  time1(ii) = ax.StartTime(4)*60*60 + ax.StartTime(5)*60 + ax.StartTime(6);
  ii=ii+1;
end
time1 = time1 - time1(1);

if use_bl_correction
  xx = (1:size(y_RS,1))';
  idx = [1:3000, size(y_RS,1) - 3000] ;
  for ii=1:size(y_RS, 2)
    [A] = polyfit(xx(idx), y_RS(idx,ii), 2);
    y_RS(:,ii) = y_RS(:,ii) - polyval(A, xx);
  end
end

res.x = x_RS;
res.y = y_RS;
res.time = time1;

dispstat(sprintf('Data "%s" are loaded.', the_path), 'keepthis');
disp(sprintf('SRC_Frequency: %s', dsc.SRC_Frequency));

if isfield(pars, 'fig') && ~isempty(pars.fig)
  figure(pars.fig); clf
  
  n = size(y_RS, 2);
  
  if n <= 35, step = 1; else step = 2; end
  
  the_max = max(y_RS(:));
  
  p=1;
  for ii=1:step:min([n, 35*step])
    if ii <= n
      subplot(5,7,p);
      p = p+1;
      plot(x_RS, y_RS(:,ii));
      axis([-30,30, -Inf, the_max])
      title(sprintf('%3.0f s', time1(ii)))
    end
  end

end