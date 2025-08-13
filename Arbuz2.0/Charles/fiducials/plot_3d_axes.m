function plot_vectors_3d(axlen)

cameratoolbar
plot3([-axlen axlen], [0 0] ,[0 0])
hold on
plot3([0 0],[-axlen axlen], [0 0])
plot3([0 0],[0 0],[-axlen axlen])
text(axlen,0,0,'X');
text(0,axlen,0,'Y');
text(0,0,axlen,'Z');
