% Function to send binary data to uart2bus via uart interface , following the protocol described
% in http://opencores.org/project,uart2bus by 
%    Litochevski, Moti &
%    MULLER, Steve
% written by Li Sun 10/27/12
% INPUT
%   s - com port 
%   data - bytes
% OUTPUT
%   out - 0 ok 1 not
function out = pgensendprog(s, data, address)
    data = data';
    data = data(:);
    
    len = length(data);
    out = 1;
    
    if len + address > 65536
        error('Error: Data larger than memory');
    end

    if len == 0
        error('Error: Data empty');
    end
    
    N = ceil(len/256);
    for i = 1:N-1
        senddata256(s, data((i-1)*256+1:i*256), address+(i-1)*256);
    end
    
    senddata256(s, data((N-1)*256+1:end), address+(N-1)*256);
    
    pause(0.1+len/16000);
    
    if s.BytesAvailable ~= N
        return;
    else
        for i = 1:N
            if fread(s, 1, 'uint8') ~= uint8(90) % ASK = '0x5A' 
                return;
            end 
        end
    end
    
    out = 0;
end


function senddata256(s, data, address)

add_hi = floor(address/256);
add_lo = mod(address,256);

write_command = 33; % 0010-0001 write command with NOP
    
cmd_byte = uint8([0, write_command, add_hi, add_lo, length(data), data(:)']);

fwrite(s, cmd_byte);
    
end
