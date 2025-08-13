% Function generate pulse program based on pulse timing information
% Inputs:
%     pulse_timing - Nx3 pulse_timing information
%                       Column 1: channel number, 1~16
%                       Column 2: pulse on, in units of 2ns
%                       Column 3: pulse off, in units of 2ns
%     polarity - polarity of channels when pulse is off
%                   for example:   polarity = bin2dec('0000000000000100')
%                   means thrid channel is high when pulse is off, and low
%                   when pulse is on, all other channels are the opposite
%                   default:0
% Output:
%     p - pulse program to be sent by senddata.m
function p = pgenpulseprog(pulse_timing, polarity)

    if nargin == 1
        polarity = 0;
    end
    
    MAXTIME = 2^32;
    N = size(pulse_timing, 1);
    
    pt2 = zeros(N, 4);
    
    for i = 1:N
        pt2(i*2-1, 1:3) = [ pulse_timing(i,2), pulse_timing(i,1), 1];
        pt2(i*2, 1:3) = [pulse_timing(i,3), pulse_timing(i,1), 0];
        if pulse_timing(i,2) >= pulse_timing(i,3) || pulse_timing(i,2) > MAXTIME || pulse_timing(i,3) > MAXTIME
            error('ERROR:generatePulseProgram.m::pulses timing error');
        end
    end
    
    [~, idx] = sort(pt2(:,1), 1);
    pt3 = pt2(idx,:);
    
    current_pulse = zeros(16,1);
    
    for i = 1:2*N
        current_pulse = setChenStatus(current_pulse, pt3(i, 2), pt3(i,3));
        pt3(i,4) = get_bit_from_chan_status(current_pulse);
    end
    
    non_identical_idx = [1, find(pt3(1:end-1, 4)' ~= pt3(2:end, 4)') + 1];
   
    pt4 = pt3(non_identical_idx, [1,4]);

    non_identical_idx2 = [find(pt4(1:end-1, 1)' ~= pt4(2:end, 1)'), size(pt4,1)];
    
    pt4 = pt4(non_identical_idx2, :);
    
    % inital command : [start_address (13 bits pad to 16), end_address (13 bits pad to 16), 0x00, 0x00, non_trigger_output(16 bits)]
    % this previously was just: p = [1, 0, int2uint8array(pt4(end,1),4) , int2uint8array(polarity,2)];
    if mod(pt4(end,1), 2) == 0
        p = [1, 0, int2uint8array(pt4(end,1),4) , int2uint8array(polarity,2)];
    else
        p = [1, 0, int2uint8array(pt4(end,1)+1,4) , int2uint8array(polarity,2)];
    end
        
    if mod(pt4(1,1), 2) == 0
        p = [p; int2uint8array(pt4(1,1), 4), int2uint8array(polarity,2), int2uint8array(polarity,2)];
    else
        p = [p; int2uint8array(pt4(1,1) + 1, 4), int2uint8array(polarity,2), int2uint8array(bitxor(polarity, pt4(1,2)),2)];
    end
    
    
    for i = 2:size(pt4,1)
        if mod(pt4(i,1), 2) == 0
            if pt4(i,1) ~= pt4(i-1,1)+1
                p = [p; int2uint8array(pt4(i,1), 4), int2uint8array(bitxor(polarity, pt4(i-1,2)),2), int2uint8array(bitxor(polarity, pt4(i-1,2)),2)];
            end
        else
            p = [p; int2uint8array(pt4(i,1) + 1, 4), int2uint8array(bitxor(polarity, pt4(i-1,2)),2), int2uint8array(bitxor(polarity, pt4(i,2)),2)];
        end
    end
    
    
end


function updates_chan_status = setChenStatus(chan_status, channel, event)
    updates_chan_status = chan_status;
    if event == 1
        updates_chan_status(channel) = chan_status(channel) +1;
    else
        updates_chan_status(channel) = chan_status(channel) -1;
    end
end

function chan_bit = get_bit_from_chan_status(chan_status)
    chan_bit = uint16(sum(bitset(uint16(0), find(chan_status>0), 1)));
end

