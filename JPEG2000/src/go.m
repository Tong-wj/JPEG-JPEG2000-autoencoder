clear;clc;close all
load 'lean.mat';
RGB=lena;
% RGB = imread('kodim23.png');
file=cell(16,1);            % 定义存储图像分块的元胞数组
[x1]=coltrans(RGB);         % 颜色变换和电平归一化
y1=db97(x1,3);              % 小波变换
tile=split_4(y1);           % 图像分块并存入元胞数组
a_new=EBCODE_main(tile);    % 进行EBCOT编码与Huffman编码

% 分块处理，这里是将256*256分为8*8以提高运算速度
tile=a_new;
t=1;
for i=1:4
    for j=1:4
     JPEG(64*j-63:64*j,64*i-63:64*i,:)=tile{t};
     t=t+1;
    end
end
x2=db97_re(JPEG,3);         % 小波反变换
y2=coltrans_re(x2);         % 颜色反变换

% 结果显示
subplot(2,2,1)
imshow(lena);
title('原始图像');
subplot(2,2,2)
imshow(x1);
title('颜色转换后的图像');
subplot(2,2,3)
imshow(y1);
title('小波变换后转换后的图像');
subplot(2,2,4)
imshow(y2);
title('压缩还原后的图像');