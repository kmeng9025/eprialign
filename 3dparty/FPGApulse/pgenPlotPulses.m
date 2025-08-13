function pgenPlotPulses(p, scale)

N = size(p,1)-1;

t = zeros(1, 2*N+1);
c = zeros(16, 2*N+1);

t(1) = 0;

for i = 1:N
    t(i * 2    ) = uint8array2int(p(i + 1, 1:4))-1;
    t(i * 2 + 1) = uint8array2int(p(i + 1, 1:4)  );
    c(:, i * 2 - 1) = bitget(uint8array2int(p(i + 1, 5:7 )), (1:16)');
    c(:, i * 2    ) = bitget(uint8array2int(p(i + 1, 7:8 )), (1:16)');    
end

c(:,2*N+1) = bitget(uint8array2int(p(1, 7:8)), (1:16)');

t(2*N+2) = uint8array2int(p(N + 1, 1:4 )) + 10;
c(:,2*N+2) = bitget(uint8array2int(p(1, 7:8)), (1:16)');


if nargin > 1 && scale == 1
    hold off
    for i = 1:16
        stairs(t*2, c(i,:)-i*1.5+16, '-', 'LineWidth',2);
        hold on     
        text(0, 16.7-i*1.5, num2str(i), 'FontSize', 12);
    end
    title('Pulse Program Timing Diagram');
else
    hold off
    for i = 1:16
        stairs(c(i,:)-i*1.5+16, '-', 'LineWidth',2);
        hold on
        text(1, 16.7-i*1.5, num2str(i), 'FontSize', 12);
    end
    set(gca, 'XTickLabel', cellstr(num2str((t*2)'))');
    title('Pulse Program Timing Diagram (TIME NOT TO SCALE)');
end
xlabel('Time (ns)');
ylim([-10, 18])
set(gca,'XGrid','on')
set(gca,'YTickLabel','')

end

