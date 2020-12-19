function [y,step]=quantization(x,subband)
%% function quantization
% Description:
% 该函数用于对输入数据进行多种步长的量化
% x：输入数据
% subband：数据所在子带信息
% y：输出量化后的数据
% step：输出所选择的量化步长
%%
switch subband      % 根据所处不同的子带，选择不同的增益
    case 0 %% LL
        gainb=0;
    case 1 %% LH
        gainb=1;
    case 2 %% HL
        gainb=1;
    case 3 %% HH
        gainb=2;
end
if max(max(x))>1    % 根据数据的大小范围，选择不同的RI
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