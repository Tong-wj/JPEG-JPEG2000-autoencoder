function [K]=coltrans_re(x)
Y = x(:,:,1);
U = x(:,:,2);
V = x(:,:,3);
RGB1 = zeros(size(x));
%颜色反变换
RGB1(:,:,1) = Y + 1.14 * V;
RGB1(:,:,2) = Y - 0.39 * U - 0.58 * V;
RGB1(:,:,3) = Y + 2.03 * U;
%反电平位移
RGB1=RGB1*256;
RGB1=RGB1+128;
K=uint8(RGB1);
end

