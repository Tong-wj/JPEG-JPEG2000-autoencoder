clear;clc;close all
load 'lean.mat';
RGB=lena;
% RGB = imread('kodim23.png');
file=cell(16,1);            % ����洢ͼ��ֿ��Ԫ������
[x1]=coltrans(RGB);         % ��ɫ�任�͵�ƽ��һ��
y1=db97(x1,3);              % С���任
tile=split_4(y1);           % ͼ��ֿ鲢����Ԫ������
a_new=EBCODE_main(tile);    % ����EBCOT������Huffman����

% �ֿ鴦�������ǽ�256*256��Ϊ8*8����������ٶ�
tile=a_new;
t=1;
for i=1:4
    for j=1:4
     JPEG(64*j-63:64*j,64*i-63:64*i,:)=tile{t};
     t=t+1;
    end
end
x2=db97_re(JPEG,3);         % С�����任
y2=coltrans_re(x2);         % ��ɫ���任

% �����ʾ
subplot(2,2,1)
imshow(lena);
title('ԭʼͼ��');
subplot(2,2,2)
imshow(x1);
title('��ɫת�����ͼ��');
subplot(2,2,3)
imshow(y1);
title('С���任��ת�����ͼ��');
subplot(2,2,4)
imshow(y2);
title('ѹ����ԭ���ͼ��');