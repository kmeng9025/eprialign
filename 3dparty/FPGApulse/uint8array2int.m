% convert an integer d to an array of uint8 of size n, lest significant
% first

function a = uint8array2int(d)
    
    a = 0;
    
    for i = 1:size(d,2)
        a = int32(d(i))*256^(i-1) + a;
    end;

end