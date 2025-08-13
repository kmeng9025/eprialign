function pO2 = imrt_SUV2PO2(SUV, parameters)

% model 2

alphac = 10.9;
betac = 10.7;
gammac = 2.5;

k = (alphac - SUV)/betac;

pO2 = gammac .* k ./(1-k);

pO2 (pO2 < -15 | pO2 > 150) = -100;

return;


%% Model 1

alphac = 4;
gammac = 0.15;
betac = 5;
testpO2 = 0:60; testSUV = 4.1:0.4:8; 
figure(10); clf; plot(testpO2, alphac + betac ./ (1 + exp(gammac*(testpO2 - 10)))); hold on;
plot(10 + log(betac ./ (testSUV - alphac) -1)/ gammac, testSUV, 'o');

%% Model 2

alphac = 10.9;
betac = 10.7;
gammac = 2.5;
testpO2 = 0:60; testSUV = 1:0.4:8; 
figure(10); clf; plot(testpO2, alphac - (betac * testpO2) ./ (gammac + testpO2)); hold on
k = (alphac - testSUV)/betac;
plot(gammac .* k ./(1-k), testSUV, 'o');
