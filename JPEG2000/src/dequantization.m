function y=dequantization(x,step)
%% function dequantization
% Description:
% 该函数用于对输入数据进行相应步长的反量化
% x：输入数据
% step：量化步长
% y：输出反量化后的数据
%%
y=sign(x).*(abs(x)+0.5)*step;
end