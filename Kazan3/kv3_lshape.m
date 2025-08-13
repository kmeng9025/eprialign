function res = kv3_lshape(x,x0,fwhmGauss,fwhmLorentz)

res = voigtian(x,x0,[abs(fwhmGauss),abs(fwhmLorentz)]);