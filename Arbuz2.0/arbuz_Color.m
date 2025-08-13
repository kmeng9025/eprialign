function [MLABcolor, HTMLcolor] = arbuz_Color(the_palette, nColor)

p1 = {'FF0000', '0101DF', '01DF01', 'FFFF00', 'FF00FF', 'FF8000', 'BFFF00', '00FFFF', ...
  '9A2EFE', '000000', 'F79F81', 'FF0040', '8181F7', '0B610B', '8258FA', 'D0A9F5', ...
  '8A0808', '81F7BE'};
n = fix(nColor);
n = min([length(p1), n]);
n = max(1, n);

the_color = p1{fix(n)};

MLABcolor = [hex2dec(the_color(1:2)), hex2dec(the_color(3:4)), hex2dec(the_color(5:6))] ./ 255;
HTMLcolor = ['#',p1{fix(n)}];

