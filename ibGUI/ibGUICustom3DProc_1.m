function Data = ibGUICustom3DProc_1(Data)

sz = size(Data);

x = 1:sz(1);
y = 1:sz(2);
z = 1:sz(3);

[X, Y, Z] = meshgrid(x, y, z);

m = (sz(1)+1) / 2;
x1 = x; %  + (x- m).^5 * 0.0000006;
y1 = y - (y- m).^3 * 0.0006;
z1 = z;

figure(5); plot(x,y1, x, x)
[X1, Y1, Z1] = meshgrid(x1, y1, z1);

DataPrime = interp3(X, Y, Z, Data, X1, Y1, Z1);
% DataPrime = interp3(X1, Y1, Z1, Data, X, Y, Z);

ibGUI(DataPrime)
