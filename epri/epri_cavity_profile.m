function profile = epri_cavity_profile(rx, st_fft, pars)
center_offset = safeget(st_fft, 'profile_center_offset', 0.0);
profile_threshold = safeget(st_fft, 'profile_threshold', 0.05);
profile_threshold_smooth = safeget(st_fft, 'profile_threshold_smooth', 0.0);
suppress_zero = true;

figure_number = 1;

discretex = [];
discretey = [];

% Cavity profile correction
switch safeget(st_fft, 'profile_correction', 'off')
    case 'library'
        fpath = safeget(st_fft, 'library_location', 'z:\CenterMATLAB\calibration\cavity_profile');
        template = ['cavity_profile_',pars.Resonator,'_'];
        res = dir(fullfile(fpath, [template,'*.mat']));
        for ii=1:length(res)
            [~,fname] = fileparts(res(ii).name);
            Qstring = fname(length(template)+1:end);
            Qstring(Qstring == 'Q') = ' ';
            Qstring(Qstring == 'p') = '.';
            profiles{ii}.Q = str2double(Qstring);
            profiles{ii}.profile = res(ii).name;
        end
        Qarray = cellfun(@(x) x.Q, profiles);
        [~, min_idx] = min(abs(Qarray - pars.Q));
        st_fft.profile_correction = 'imager_profile';
        st_fft.profile_file = fullfile(fpath, profiles{min_idx}.profile);
        
        fprintf('epri_cavity_profile: loading %s\n', st_fft.profile_file);
        rawprofile = epri_cavity_profile(rx, st_fft, pars);
        figure_number = -1;
        
    case 'lorentz'
        fwhh = safeget(st_fft, 'profile_fwhh', 20.0);
        rawprofile = epri_lshape(rx, center_offset, [0, fwhh(1)]);
        for ii=2:length(fwhh)
            rawprofile = rawprofile .* epri_lshape(rx, center_offset, [0, fwhh(ii)]);
        end
        rawprofile = rawprofile / max(rawprofile);
        
    case 'imager_profile'
        [discretex,discretey]=load_profile(safeget(st_fft, 'profile_file', ''));
        rawprofile = interp1(discretex + center_offset, discretey, rx, 'pchip');
        rawprofile = rawprofile./max(rawprofile);
        
    case 'cos3fitted'
        [discretex,discretey]=load_profile(safeget(st_fft, 'profile_file', ''));
        f   = @(x) x(1) * cos(3.14*(discretex - x(3))*x(2)).^3;
        fit = @(x) sqrt(sum((discretey - f(x)).^2));
        x =  [1.0, 0.1, 0];
        x = fminsearch(fit, x);
        x = fminsearch(fit, x);
        rawprofile =  x(1)*cos(3.14*(rx-x(3))*x(2)).^3;
    
    case 'gaussfitted'
        [discretex,discretey]=load_profile(safeget(st_fft, 'profile_file', ''));
        if suppress_zero
            idx = abs(discretex) > 0.15;
            dx = discretex(idx);
            dy = discretey(idx);
        end
        f   = @(x) x(1) * exp((dx - x(3)).^2*x(2));
        fit = @(x) sqrt(sum((dy - f(x)).^2));
        x =  [1.0, 0.1, 0];
        x = fminsearch(fit, x);
        x = fminsearch(fit, x);
        rawprofile =  x(1) * exp((rx - x(3)).^2*x(2));
        
    case 'sincfitted'
        [discretex,discretey]=load_profile(safeget(st_fft, 'profile_file', ''));
        f   = @(x) x(1) * sinc((discretex - x(3))*x(2)).^3;
        fit = @(x) sqrt(sum((discretey - f(x)).^2));
        x =  [1.0, 0.1, 0];
        x = fminsearch(fit, x);
        x = fminsearch(fit, x);
        rawprofile = x(1) * sinc((rx - x(3))*x(2));
        
    case {'off', 'none'}
        rawprofile = ones(size(rx));
end

rawprofile = real(rawprofile);
scaling = max(rawprofile);
profile = rawprofile./scaling;            
profile(profile < profile_threshold) = profile_threshold;

if figure_number >= 1
    figure(figure_number); clf; hold on;
    FOV = safeget(st_fft, 'FOV', 20);
    idx =  rx >= -FOV/2 & rx <= FOV/2 ;
    plot(rx(idx), profile(idx));
    if ~isempty(discretey)
        idx = discretex >= -FOV/2 & discretex <= FOV/2;
        plot(discretex(idx),discretey(idx)/scaling,'o');
    end
    axis([-FOV/2, FOV/2,0,1])
    title('Cavity Profile');
    grid on
end

function [x,y] = load_profile(fname)
if ~isempty(fname) && exist(fname, 'file')
    s = load(fname);
    [~, idx] = unique(s.Frequency);
    y = s.Amplitude(idx) / max(s.Amplitude(idx));
    x = s.Frequency(idx);
else
    error('Profile is not loaded.');
end

function res = sinc(x)
idx = x == 0;
res = ones(size(x));
res(~idx) = sin(x) ./ x;

