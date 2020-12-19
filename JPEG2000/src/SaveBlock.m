function y=SaveBlock(widthBlock,heightBlock,level,subband,Block)
%% function SaveBlock.m
% Descritption：
% 该函数用于保存数据块的信息，构成一个结构体
% widthBlock：数据块的宽度
% heightBlock：数据块的高度
% level：数据块的级数信息
% subband：数据块的子带信息
% Block：数据块的数值
%%
y.width=widthBlock;
y.height=heightBlock;
y.level=level;
y.subband=subband;
y.data=Block;
end