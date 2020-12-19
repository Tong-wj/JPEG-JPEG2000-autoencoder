function [y,h,v,d]=GetSignificantNeighbors(m,width,height,i,j)
%% function GetSignificantNeighbors.m
% Description:
% 该函数用于获取输入位置所对应的邻居的重要性信息
% m：数据集
% width：数据集的宽度，即列数
% height：数据集的高度，即行数
% i,j：待考察的位置坐标
%%
y=0;v=0;h=0;d=0;
if i>1 && m(i-1,j)
	v=v+1;
	y=1;
end
if i < height-1 && m(i+1,j)
	v=v+1;
	y=1;
end

if j>1 && m(i,j-1)
	h=h+1;
	y=1;
end
if j<width-1 && m(i,j+1) 
	h=h+1;
	y=1;
end
if i>1 && j>1 && m(i-1,j-1)
	d=d+1;
	y=1;
end
if i>1 && j<width-1 && m(i-1,j+1)
	d=d+1;
	y=1;
end
if i<height-1 && j>1 && m(i+1,j-1)
	d=d+1;
	y=1;
end
if i<height-1 && j<width-1 && m(i+1,j+1)
	d=d+1;
	y=1;
end
end