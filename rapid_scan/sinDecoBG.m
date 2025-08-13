function [h, A, B, fp_corr_total, rs_out, bg_out]=sinDecoBG(par)

%% HELP
% ! NOTE that interchanging of I (real) and Q (imaginary) cables is analogous
% to complex conjugation of the complex rapid scan signal.
% To test cable position one can measure two spectra with a center field (CF)offset.
% After deconvolution the position of EPR line must shift to the lower fields if CF is increased.
% It is similar to the CW EPR would do shift with a CF offet.
%Input Pars   --------------------
% par.hm     [G] Peak-to-peak modulation amplitude
% par.Vm     [Hz] Modulation frequency
% par.rs     Experimantal Rapid scan signal, must have at least one full cycle
% par.dt     dt -  time base (sampling period) for signal stored in array [s]
% par.up     up must be '1' if the first half-scan has up direction, '-1' for down-scan.
% par.ph     RF (MW) phase correction [degrees]
% par.fp     First point of the cycle to be deconvolved in units of degrees [0 360]
%            For experimental data fp has to be adjusted,so that A and B (spectra for up and down scans) coinside.
% par.bw;    Currently is not used
%            [Hz] Expected signal BW for pre-filtering, should be slighty
%            larger than the resonator BW
% par.fwhm;  [G] Estimated full width at the half height of the narrowest line in the spectrum.
%            It is used for spectrum filtering with a gaussain profile with width =par.fwhm/10
% par.msg    '1' to show warning messages
%            '0' switches off warning messages to speed up data processing
% par.fig=1; '1' is used for adjustment of par.fp to match up and dwon spectra;
%            '0' does not refresh the figure and is used to speed up the
%             deconvolution program for automated data processing.
% par.method='fast';  'default is fast', for very sensetive experiments
%            you may try  'slow'
% fp_corr    estimate for fp correction
% Output Pars ---------------------------
%  h - magnetic field [G]
%  A & B - deconvolved spectra for up and down scan,
%  A and B must be summed up after the fp is adjusted
% ------------------------------------------------------
% Version 2014_2; August 2014
% Mark Tseytlin; University of Denver;
% mark.tseytlin@nsm.du.edu | mark.tseytlin@du.edu
% ----------------------------------------------------
%% Input parameters
gamma=1.7608e7;
g2f=gamma/(2*pi); % = 2.8024e6


up=par.up;   %  up must be '1' for up or '-1' for down scan.
ph=par.ph;   %  phase correction [degrees]
fwhm=par.fwhm;
show=par.fig;         % 1 to show Figure, 0 not to show; should be used to speed up computaion
method=par.method;  % method of interleaving '1' slower-better, '2'- faster
N_iter=safeget(par, 'N_iter', 2);
fp=par.fp;   %  First point of the cycle to be deconvolved in units of the
Hm=par.hm;   %  Peak-to-peakp modulation amplitude [G]
Vm=par.Vm;   %  Modulation frequency    [Hz]
fp_corr_total=0;
for iterate_fp=1:N_iter
    tb=par.dt;   %  dt -  time base (sampling period) for signal stored in array [s]
    rs=par.rs;   %  Rs signal    
    %% Derived parameters
    M=length(rs);
    ss=size(rs);
    if ss(1)>ss(2); rs=transpose(rs); end
    t=(0:(M-1))*tb;       % Time vector ( raw data)
    Vmax=g2f*Hm;          % Max possible RS signal frequency
    Ns=2*ceil(Vmax/Vm);   % Min number of points in the frequency domain(= the time domain)
    P=1/Vm;               % Scan period
    ts=(0:(Ns-1))*(P/Ns); % time vector for final filtering & interpolation
    Fmax=1/(2*tb);        % Max frequency to be sampled without aliasimg
    ratio=Vmax/Fmax;      % sampling ratio must be <1
    Wm=2*pi*Vm;
    
    % outputs: t, Vmax, ts, P
    
    %% Error & Warning check
    Nc=round(1/(tb*Vm));      % Points per period
    Nfc=floor(M/Nc);           % Number of full cycles
    if safeget(par, 'msg', 0)
        if Nfc==0; disp('ERROR: less than a full cycle'); error('RS signal must have at least one full cycle'); return; end;
        if ratio>1; disp('Warning: Sampling rate may not be sufficient'); end;
    end
    
    %% Interleaving data to one periodic cycle
    try
    rs_i=InterleavingCycles(t,ts,rs,Vm,Nfc,Ns,Wm,method);
    catch err
      err
    end
    %plot(t,rs*1.05,ts+0*P,rs_i);
    
    %break
    %% Phase correction
    rs_ii=rs_i*exp(1i*ph/180*pi);
    
    %% Position of 1st point correction
    shift=round(fp/360*length(rs_ii));   % Circular shift to find 1st point
    rs_iii=circshift(rs_ii',shift)';
    rs_out=rs_iii;
    %% Driving function
    rs=rs_iii;
    t=ts;
    tb=ts(2); % new time base
    Nc=Ns;
    %plot(t,rs);
    WF=-up*cos(2*pi*Vm*t);
    W=gamma*Hm/2*WF;            % waverform
    dr=exp(-1i*cumsum(W)*tb); % driving function for 1 cycle
    
    %% Separation Up from Down
    [v RS]=fftM(t,rs);
    if up>0
        in=v>0;
        a=ifft(ifftshift(RS.*in)); % up scan
        jn=v<0;
        b=ifft(ifftshift(RS.*jn)); % up scan
        aa=a(1:Ns/2);
        bb=b(Ns/2+1:Ns);
        bga=a(Ns/2+1:Ns);
        bgb=b(1:Ns/2);
    else
        in=v<0;
        a=ifft(ifftshift(RS.*in)); % up scan
        jn=v>0;
        b=ifft(ifftshift(RS.*jn)); % up scan
        aa=a(1:Ns/2);
        bb=b(Ns/2+1:Ns);
        bga=a(Ns/2+1:Ns);
        bgb=b(1:Ns/2);
    end
    t1=t(1:Ns/2);
    t2=t(Ns/2+1:Ns);
    %plot(t1,imag(aa),t2,imag(bb));
    
    %% BG removal
    % 1st half
    x=2*pi*Vm*t1;
    c1=cos(x);
    c2=cos(2*x);
    s1=sin(x);
    s2=sin(2*x);
    
    for kk=1:2 % real and imaginary
        if kk==1
            r=H_amplit(t2,real(bga),Vm) ;
            %   sinx  sin2x  cosx  cos2x const
            % r=[xs(1) xa(1) xa(2) xs(2) xs(3)];
            bg=r(1)*s1+r(2)*s2+r(3)*c1+r(4)*c2+r(5);
        else
            r=H_amplit(t2,imag(bga),Vm) ;
            tmp=r(1)*s1+r(2)*s2+r(3)*c1+r(4)*c2+r(5);
            bg=bg+1i*tmp;
        end
    end
    aaa=aa-bg;
    bgA=bg;
    %pltc(aaa)
    %% 2nd half
    x=2*pi*Vm*t2;
    c1=cos(x);
    c2=cos(2*x);
    s1=sin(x);
    s2=sin(2*x);
    
    for kk=1:2 % real and imaginary
        if kk==1
            r=H_amplit(t1,real(bgb),Vm) ;
            %   sinx  sin2x  cosx  cos2x const
            % r=[xs(1) xa(1) xa(2) xs(2) xs(3)];
            bg=r(1)*s1+r(2)*s2+r(3)*c1+r(4)*c2+r(5);
        else
            r=H_amplit(t1,imag(bgb),Vm) ;
            tmp=r(1)*s1+r(2)*s2+r(3)*c1+r(4)*c2+r(5);
            bg=bg+1i*tmp;
        end
    end
    bbb=bb-bg;
    bgB=bg;
    %pltc(bbb);
    % %%
    bg_out=rs-[aaa bbb];
    %break
    %% 1th BaseLine Correction
    %aaa=zeroLine(aaa,0.05);
    %bbb=zeroLine(bbb,0.05);
    %subplot(2,1,1); pltc(aaa);
    %subplot(2,1,2); pltc(bbb);
    %% Deco for A
    drA=dr(1:Ns/2);
    aaaa=aaa.*drA;
    [v A]=fftM(t,aaaa);
    [v D]=fftM(t,drA);
    A=A./(D);
    %% Deco for B
    drB=dr(Ns/2+1:Ns);
    bbbb=bbb.*drB;
    [v B]=fftM(t,bbbb);
    [v D]=fftM(t,drB);
    B=B./(D);
    h=v*2*pi/gamma;
    in=abs(h)<Hm/2;
    A=A(in);
    B=B(in);
    h=h(in);
    %% Post -filtering
    filter=mygaussian(h,fwhm/10);
    sm=sum(filter);
    if sm>0
        filter=filter/sum(filter);
        A=conv(A,filter,'same');
        B=conv(B,filter,'same');
        %else
        %   disp('Gaussian post-filter is ignored')
    end
    
    %plot(h,real(BB),h,real(B))
    A=zeroLine(A,0.05);
    B=zeroLine(B,0.05);
    A=imag(fliplr(A));
    B=imag(fliplr(B));
    
    %% Correct Scan Phase  Estimation
    if N_iter>1
        Af=interpft(A,length(A)*4);
        Bf=interpft(B,length(B)*4);
        hi=linspace(h(1),h(end),length(Af));
        filter_i=mygaussian(hi,fwhm);
        %filter_i=filter-min(filter);
        sm=sum(filter);
        if sm>0
            filter_i=filter_i/sum(filter_i);
            A1=gradient(conv(Af,filter_i,'same'));
            B1=gradient(conv(Bf,filter_i,'same'));
        end
        [min_A max_A max_ha min_ha mid_ha]=min_max(hi,A1);
        [min_B max_B max_hb min_hb mid_hb]=min_max(hi,B1);
        h_ab=mean([mid_ha mid_hb]);
        dh_ab=[mid_ha mid_hb]-h_ab;
        rate_max=pi*par.hm*par.Vm;
        time_shift=dh_ab/rate_max;
        fp_corr=time_shift*par.Vm*360;
        %plot(hi,A1,hi,B1); %hold on;
        fp_corr_total=fp_corr_total+fp_corr(1);
        fp=fp-fp_corr(1);
    else
        fp_corr=fp_corr_total;
    end
end

%% Show Figures
if show
    figure(par.fig)
    subplot(2,2,1);
    set(gca,'FontSize',16);
    rss=par.rs;
    tt=(1:length(rss))*tb;
    plot(tt*1E6,real(rss),tt*1E6,imag(rss));
    xlabel 'Time, us'
    axis tight;
    title 'Raw data'
    %%
    subplot(2,2,2);
    set(gca,'FontSize',16);
    plot(t*1E6,real(rs),t*1E6,imag(rs));
    xlabel 'Time, us'
    axis tight;
    title 'Averaged full RS cycle'
    %%
    subplot(2,2,[3 4]);
    
    set(gca,'FontSize',16);
    plot(h,A,h,B); %hold on;
    %plot(h,real(A),h,real(B)); hold off;
    
    axis tight;
    xlabel 'Magnetic Field, G'
    
end

