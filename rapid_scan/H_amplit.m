function r=H_amplt(xi,yi,Fm)

% Fm - scan frequency
% Nh - number of harmonics

t=2*pi*Fm*xi;

c1=cos(t);
s1=sin(t);

s2=sin(2*t);
c2=cos(2*t);


o=c1*0+1;

f=yi;
fs=(f+fliplr(f))/2;
fa=(f-fliplr(f))/2;
%if t(1)==0
%% Symmetric part
% fsTest=b1*s1+a2*c2+c*o;
% testS=sum(fsTest-fs)
%
v1=s1;
v2=c2;
v3=o;
%;
f=[v1*fs'; v2*fs'; v3*fs'];
T=[v1*v1' v1*v2' v1*v3'; v2*v1' v2*v2' v2*v3'; v3*v1' v3*v2' v3*v3'];
xs=inv(T)*f; % = [b1 a2 c]
% test=sum(xs-[b1 a2 c]')
%% Asymmetric part
% faTest=b2*s2+a1*c1;
% testA=sum(faTest-fa)

v1=s2;
v2=c1;
%
f=[v1*fa'; v2*fa'];
T=[v1*v1' v1*v2'; v2*v1' v2*v2'];
xa=inv(T)*f; % = [b2 a1]

%    sinx  sin2x cosx  cos2x const
r=[xs(1) xa(1) xa(2) xs(2) xs(3)];  % result
%z=r(1)*s1+r(2)*s2+r(3)*c1+r(4)*c2+r(5);
%plt2(yi,z)


