function iradon_verify(ideal_object,object)

% Function: test the reconstructed object and observe the precision
%
%
% ideal_object, a 3-D array, which is the standard model image.
% object, a 3-D array,which is the reconstructed object.
% there are four results for testing
% 1. color image comparison
% 2. gray image comparison
% 3. average absolute error and average relative error
% 4. comparison of profile of the central line of the central slice of the object
%
%
%
% Author: ZHIWEI QIAO
% Center for EPR imaging in vivo physiology
% University of Chicago, JULY,2013
% Contact: zqiao1@uchicago.edu
%

error_object=abs(object-ideal_object);
a=size(object);
number_of_finalimage=a(1);

figure;%****************method 1********************
subplot(2,2,1)
imagesc(squeeze(ideal_object(:,:,number_of_finalimage/2)));
title('the ideal object');
subplot(2,2,2)
imagesc(squeeze(object(:,:,number_of_finalimage/2)));
title('the reconstructed object');
subplot(2,2,3)
imagesc(error_object(:,:,number_of_finalimage/2),[0 max(max(max(ideal_object)))]);
title('the error object');

figure;%****************method 2********************
subplot(2,2,1);
imshow(squeeze(ideal_object(:,:,number_of_finalimage/2)));
title('the ideal object');
subplot(2,2,2);
imshow(squeeze(object(:,:,number_of_finalimage/2)));
title('the reconstructed object');
subplot(2,2,3);
imshow(squeeze(error_object(:,:,number_of_finalimage/2)),[0 max(max(max(ideal_object)))]);
title('the error object');


%****************method 3********************
average_abs_error=mean(mean(mean(abs(object-ideal_object))));
average_relative_error=average_abs_error/mean(mean(mean(ideal_object)));
fprintf('the average absolute error is %f \n',average_abs_error);
fprintf('the average relative error is %6.2f%% \n',average_relative_error*100);

%****************method 4********************
ideal_profile=squeeze(ideal_object(number_of_finalimage/2,:,number_of_finalimage/2));
my_profile=squeeze(object(number_of_finalimage/2,:,number_of_finalimage/2));
figure;
plot(ideal_profile,'r');
hold on;
plot(my_profile,'b');
legend('the ideal profile','my profile');
xlabel(strcat('the average relative error is ', num2str(average_relative_error)));
title(strcat('the average absolute error is ', num2str(average_abs_error)));
end

