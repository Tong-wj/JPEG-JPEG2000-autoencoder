function [y,x_new,num]=InitializeReading(x,num)
%% function InitializeReading.m
% Description��
% �ú������ڶ��ļ��ṹ�Ķ�ȡ���г�ʼ��
% x������Ľṹ��
% num�����ڸ��¶�ȡ��λ��
% y����ǩ����ʵûɶ�ã�
% x_new���Զ�ȡ������ʼ���Ժ�Ľṹ��
%%
x_new=x;
x_new.quedan=0;
x_new.input=0;
x_new.input=x_new.file(num);
num=num+1;
x_new.quedan=8;
y=0;
end