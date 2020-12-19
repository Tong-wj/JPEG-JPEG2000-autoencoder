function [y,step]=quantization(x,subband)
%% function quantization
% Description:
% �ú������ڶ��������ݽ��ж��ֲ���������
% x����������
% subband�����������Ӵ���Ϣ
% y����������������
% step�������ѡ�����������
%%
switch subband      % ����������ͬ���Ӵ���ѡ��ͬ������
    case 0 %% LL
        gainb=0;
    case 1 %% LH
        gainb=1;
    case 2 %% HL
        gainb=1;
    case 3 %% HH
        gainb=2;
end
if max(max(x))>1    % �������ݵĴ�С��Χ��ѡ��ͬ��RI
    RI=2;
elseif max(max(x))>1e-2
    RI=1;
else
    RI=-2;
end
Rb=RI+gainb;
step=(1+8/(2^11))*2^(Rb-7);
y=sign(x).*floor(abs(x)/step);
end