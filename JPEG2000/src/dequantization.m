function y=dequantization(x,step)
%% function dequantization
% Description:
% �ú������ڶ��������ݽ�����Ӧ�����ķ�����
% x����������
% step����������
% y������������������
%%
y=sign(x).*(abs(x)+0.5)*step;
end