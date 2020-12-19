function y=Context(h,v,d,subband)
%% function Context.m
% Description：
% 该函数用于获取位置的上下文信息
% h：水平邻居的重要性信息
% v：垂直邻居的重要性信息
% d：对角邻居的重要性信息
% subband：所在的子带信息
% y：输出的上下文的值
%%
switch subband
	case 0 % LL
        y=initializeContextLL();
        y=y(h+1,v+1,d+1);
        return;
	case 1 % LH
		y=initializeContextLL();
        y=y(h+1,v+1,d+1);
        return;
	case 2 % HL
		y=initializeContextHL();
        y=y(h+1,v+1,d+1);
        return;
	case 3 % HH
		y=initializeContextHH();
        y=y(h+1,v+1,d+1);
        return;
    otherwise
		y=-100;
        return
end
end