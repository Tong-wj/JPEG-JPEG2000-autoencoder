function [y,ho,ver]=ContextSign(h,v,significant,sign)
%% function ContextSign.m
% 该函数用于获取符号上下文信息
% h：水平邻居的重要性信息
% v：垂直邻居的重要性信息
% significant：待考察位置的重要性信息
% sign：待考察位置的符号信息
% y：输出的符号上下文值
% ho：输出每种符号上下文对应的水平邻居符号信息
% ver：输出每种符号上下文对应的垂直邻居符号信息
%%
if h==0 || v==0
    ho=0;
    ver=0;
	y=9;
    return
end
if significant(h,v+1) && significant(h+2,v+1) && sign(h,v+1) && sign(h+2,v+1)
	ho=1;
elseif significant(h,v+1) && significant(h+2,v+1) && ~sign(h,v+1) && ~sign(h+2,v+1)
	ho=-1;
else
	ho=0;
end
if significant(h+1,v) && significant(h+1,v+2) && sign(h+1,v) && sign(h+1,v+2)
	ver=1;
elseif(significant(h+1,v) && significant(h+1,v+2) && ~sign(h+1,v) && ~sign(h+1,v+2))
	ver=-1;
else
	ver=0;
end
if ho==0 && ver==0
	y=9;
    return
end
if (ho==0 && ver==1)||(ho==0 && ver==-1)
	y=10;
    return
end
if (ho==1 && ver==1)||(ho==-1 && ver==-1)
	y=13;
    return
end
if (ho==1 && ver==0)||(ho==-1 && ver==0)
	y=12;
    return
end
if (ho==1 && ver==-1)||(ho==-1 && ver==1)
	y=11;
    return
end
y=0;
return
end