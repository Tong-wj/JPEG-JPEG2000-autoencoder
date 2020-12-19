function [YUV]=coltrans(x)
%电平位移
RGB=(double(x)-128)./256;
R = RGB(:,:,1);
G = RGB(:,:,2);
B = RGB(:,:,3);
x = size(RGB,1);
y = size(RGB,2);
%颜色变换
Y = 0.299*R + 0.587*G + 0.114*B;
U = -0.147*R- 0.289*G + 0.436*B;
V = 0.615*R - 0.515*G - 0.100*B;

YUV = cat(3, Y, U, V);
end