function y=EndWriting(x)
%% function InitializeWrite.m
% Description��
% �ú������ڶ��ļ��ṹ��д�������β
% x������Ľṹ��
% y������д�����ݺ�Ľṹ��
%%
y=x;
if y.pos~=8
   y.file(end+1)=int32(y.output); 
end
end