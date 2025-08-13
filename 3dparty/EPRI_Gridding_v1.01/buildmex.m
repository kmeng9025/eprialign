cd ./common/library

mex LibConvGrid.cpp
mex LibConvGrid3.cpp
% mex LibConvInvGrid.cpp
% mex LibConvInvGrid3.cpp
mex LibConvGrid3Par.cpp -lpthread

cd ..
cd ..

