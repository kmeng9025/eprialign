function  varargout=RsSteadyStateX(T2,hm,Vm,dH,BW,RBW)% progonka y''-y=0; y(0)=1;
% [t Mx My a b];
%T2 s
%hm - mudulation amplitude, G
%Vm - modulation frequency, Hz
%dH - magnetic field offset, G
%BW - signal bandwidth, Hz
%RBW - resonator BW
if RBW>BW; RBW=BW; end;

msg = nargoutchk(2, 5, nargout);
if ~isempty(msg);
    error(msg);    
end

N=round(BW/Vm);
Nr=round(RBW/Vm);

M=2*round(1.05*N);
Mr=2*round(1*Nr);

gamma=1.76e7;
y0=hm*gamma*T2/2;
lamda=2*pi*Vm*T2; 
dx=dH*gamma*T2;
y=sqrt(-1)/2*y0;


I=sqrt(-1); % +/- ?
a=zeros(1,2*N+1); % diagonal
f=zeros(1,2*N+1); % right constants
b=zeros(1,2*N); % up
c=zeros(1,2*N); %down



x=1+dx*I;
k=N:(-1):-N;
a=lamda*k*I+x;
b(:)=y;
c(:)=y;
f(N+1)=I; % +/- ?


res=progonka3(a,b,c,f);

clear a b c f y x
a=res(1:N);           % alfa
b=res( (N+2):(2*N+1) ); % betta
a=fliplr(a);
a0=res(N+1);

MM=Mr-1;
tau=(0:MM)/MM*1/lamda*2*pi;

ii=1:N;
fltr=exp(-ii/Nr);
a=a.*fltr;
b=b.*fltr;

sum=a0;
K=repmat(1:Nr,Mr,1)'; % Mx100
TAU=repmat(tau',1,Nr)';
EX=exp(I*K.*lamda.*TAU);
sum=a0+a(1:Nr)*EX;
EX=exp(-I*K.*lamda.*TAU);
sum=sum+b(1:Nr)*EX;


%cw=imag( a(1)+b(1) );
%cw=real( a(1)-b(1) );

% mtx=conj(mtx);
Mx=T2*real(sum);
My=T2*imag(sum);

t=(0:Mr)/Mr*2*pi/lamda*T2;
t=t(1:(end-1));
varargout(1) = {t};
varargout(2) = {Mx};
varargout(3) = {My};
varargout(4) = {a};
varargout(5) = {b};
