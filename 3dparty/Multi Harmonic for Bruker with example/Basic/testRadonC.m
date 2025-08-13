clc 
clear all;
close all;
% delete radonc.dll
% mex radonD.c


 P=phantom(64);
 theta=0:10:170;
 
 R=radonD(P,theta);
 subplot(2,1,1)
 fig(R)
 
 subplot(2,1,2)
 Rt=radonD(P,theta);
 fig(Rt)
 [size(R) size(Rt)]
 
 
 