function y=EndWriting(x)
%% function InitializeWrite.m
% Description：
% 该函数用于对文件结构的写入进行收尾
% x：输入的结构体
% y：结束写入数据后的结构体
%%
y=x;
if y.pos~=8
   y.file(end+1)=int32(y.output); 
end
end