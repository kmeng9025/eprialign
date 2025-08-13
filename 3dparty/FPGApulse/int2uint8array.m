% convert an integer d to an array of uint8 of size n, lest significant
% first

function a = int2uint8array(d, n)

    a = uint8(zeros(1,n));


    for i = 1:n
        a(i) = mod(d, 256);
        d = floor(d/256);
    end

end