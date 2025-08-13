function Uim=regularized_recon(AhA,Ahb,Nbins,lambda,Uim)

L=numel(Uim)/Nbins^3;
vec=@(x)x(:);

Gim=zeros(Nbins,Nbins,Nbins,L,3);
Gim(1,1,1,:,:)=1;
Gim(end,1,1,:,1)=-1;
Gim(1,end,1,:,2)=-1;
Gim(1,1,end,:,3)=-1;
Gf=fft(fft(fft(Gim,[],1),[],2),[],3);

Grad=@(Uim)ifft(ifft(ifft(Gf.*fft(fft(fft(...
  repmat(reshape(Uim,Nbins,Nbins,Nbins,[]),[1 1 1 1 3])...
  ,[],1),[],2),[],3),[],1),[],2),[],3);
Gradh=@(Q)ifft(ifft(ifft(sum(conj(Gf).*fft(fft(fft(Q ...
  ,[],1),[],2),[],3),5),[],1),[],2),[],3);
GhG=@(Uim)vec(ifft(ifft(ifft(sum(abs(Gf).^2.*fft(fft(fft(...
  repmat(reshape(Uim,Nbins,Nbins,Nbins,[]),[1 1 1 1 3])...
  ,[],1),[],2),[],3),5),[],1),[],2),[],3));

alpha0=max(vec(sqrt(sum(sum(abs(Grad(Uim)).^2,4),5))))/10;
Uim=zeros(size(Uim));
for alpha=alpha0*logspace(0,-4,5)
  alpha
  for i=1:10
    i
    Q=Grad(Uim);
    Qw=sqrt(sum(sum(abs(Q).^2,4),5));
    Qw=max(Qw-alpha,0)./Qw;
    Qw(isnan(Qw))=0;
    Q=repmat(Qw,[1 1 1 L 3]).*Q;
    Q=Gradh(Q);
    
    Uim_new=pcg(@(x)(AhA(x)+lambda/(2*alpha)*GhG(x)),Ahb(:)+lambda/(2*alpha)*Q(:),[],20,[],[],Uim(:));
    eps = norm(Uim_new(:)-Uim(:))/norm(Uim(:))
    Uim=Uim_new;
    
    if eps < 5e-3
      break;
    end
  end
end


