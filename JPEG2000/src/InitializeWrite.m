function [y,x_new]=InitializeWrite(x)
%% function InitializeWrite.m
% Description：
% 该函数用于对文件结构进行初始化
% x：输入的结构体
% y：标签（其实没啥用）
% x_new：初始化以后的结构体
%%
x_new=x;
x_new.file=[];
x_new.output=0;
x_new.pos=8;
y=0;
end