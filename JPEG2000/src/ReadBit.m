function [y,x_new,num]=ReadBit(x,num)
%% function ReadBit.m
% Description��
% �ú������ڴ��ļ���ȡ�ض�λ�õı���λ
% x����ű��������ļ�
% num�����¶�ȡλ��
% y����ȡ���ı���λ
% x_new����ȡ����ļ�
%%
x_new=x;
if ~x_new.quedan
    x_new.input=x_new.file(num);
    num=num+1;
    x_new.quedan=8;
end
y=bitshift(bitand(int32(x_new.input),128),-7);  % ����ԭ����ʹ����ʮ�����ƽ�������
x_new.input=bitshift(x_new.input,1);
x_new.quedan=x_new.quedan-1;
end