function [y,x_new]=InitializeWrite(x)
%% function InitializeWrite.m
% Description��
% �ú������ڶ��ļ��ṹ���г�ʼ��
% x������Ľṹ��
% y����ǩ����ʵûɶ�ã�
% x_new����ʼ���Ժ�Ľṹ��
%%
x_new=x;
x_new.file=[];
x_new.output=0;
x_new.pos=8;
y=0;
end