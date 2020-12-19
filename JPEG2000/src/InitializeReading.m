function [y,x_new,num]=InitializeReading(x,num)
%% function InitializeReading.m
% Description：
% 该函数用于对文件结构的读取进行初始化
% x：输入的结构体
% num：用于更新读取的位置
% y：标签（其实没啥用）
% x_new：对读取操作初始化以后的结构体
%%
x_new=x;
x_new.quedan=0;
x_new.input=0;
x_new.input=x_new.file(num);
num=num+1;
x_new.quedan=8;
y=0;
end