classdef classCoil
   properties (Constant)
     mu0 = 4e-7*pi; % [T m A−1]
     e0 = 8.8541878128e-12; % 1/mu0/c/c [F/m]
     c = 299792458; % [m/s]
   end
   methods (Static)
     function B = Loop(I, N, R, X)
       % B [T] =  Loop(I [current, A], N [n turns], R [Radius, m], X [axial position, m])
       B = classCoil.mu0 * N * I .* R .* R / 2 ./ (R.^2 + X.^2).^1.5;
     end
     function B = CoilPair(I, N, R, A, X)
       % B [T]=  CoilPair(I [current, A], N [n turns], R [Radius, m], A [separation, m], X [axial position, m])
       % For Helmholtz coil pair Radius = Separation
       B = classCoil.Loop(I, N, R, A/2-X) + classCoil.Loop(I, N, R, A/2+X);
     end
     function B = CoilAntiPair(I, N, R, A, X)
       % B [T]=  Helmholtz(I [current, A], N [n turns], R [Radius, m], A [separation, m], X [axial position, m])
       B = classCoil.Loop(I, N, R, A/2-X) - classCoil.Loop(I, N, R, A/2+X);
     end
     function B = HelmholtzOrigin(I, N, R)
       % Field at the center of H\lmholtz pair
       % B [T]=  HelmholtzOrigin(I [current, A], N [n turns], R [Radius, m])
       B = classCoil.mu0 * 8*sqrt(5)/25 * N * I ./ R;
     end
     function B = HelmholtzError(R, X)
       % Error [ppm]=  HelmholtzError(R [Radius, m], X [distance, m])
       B = 1e6*(1-classCoil.CoilPair(1, 1, R, R, X)/classCoil.HelmholtzOrigin(1, 1, R));
     end
     function G = CoilGrad(I, N, R, X)
       % G [T/m] =  Loop(I [current, A], N [n turns], R [Radius, m], X [distance, m])
       G = classCoil.mu0 * (-3/2) * N * I .* R .* R .*X ./ (R.^2 + X.^2).^2.5;
     end
     function G = CoilPairGrad(I, N, R, A, X)
       % G [T/m]=  CoilPairGrad(I [current, A], N [n turns], R [radius, m], A [separation, m], X [axial position, m])
       G = -classCoil.CoilGrad(I, N, R, A/2-X) - classCoil.CoilGrad(I, N, R, A/2+X);
     end
     function L = InductanceCylinder(D, W)
       % L [Henry] = InductanceCylinder(D [diameter, m], W [width, m])
       L = classCoil.mu0*pi*D*D / (W + 0.45*D);
     end
     function C = Capacitance(S, d, er)
       % C [F] = Capacitance(S [square, m*m], d [gap, m], er [relative permittivity, 1 by default])
       if ~exist('er', 'var'), er = 1; end
       C =  classCoil.e0*er*S/d;
     end
     function [L, C, Z] = SlottedTube(W, Angle)
       % [L [Henry], C [Farade], Z [Ohm]] = SlottedTube(W [length, M], Angle [subtended angle of conductor, rad])
       % REF: Slotted tube resonator: A new NMR probe head at high observing frequencies
       % Rev Sci Instrum 48, 68–73 (1977)
       k = (1 - sin(Angle/2)) / (1 + sin(Angle/2)); % tan2(eta/2) eta is half openning
       kprime = sqrt(1 - k*k);
       L = classCoil.mu0*W*ellipticK(k) / ellipticK(kprime);
       C = classCoil.e0*W*ellipticK(kprime) / ellipticK(k);
       Z = sqrt((classCoil.mu0/classCoil.e0))*ellipticK(kprime) / ellipticK(k);
     end
   end
end
