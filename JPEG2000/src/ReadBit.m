function [y,x_new,num]=ReadBit(x,num)
%% function ReadBit.m
% Description：
% 该函数用于从文件读取特定位置的比特位
% x：存放比特流的文件
% num：更新读取位置
% y：读取出的比特位
% x_new：读取后的文件
%%
x_new=x;
if ~x_new.quedan
    x_new.input=x_new.file(num);
    num=num+1;
    x_new.quedan=8;
end
y=bitshift(bitand(int32(x_new.input),128),-7);  % 这里原代码使用了十六进制进行运算
x_new.input=bitshift(x_new.input,1);
x_new.quedan=x_new.quedan-1;
end