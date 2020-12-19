function [y,ho,ver]=ContextSign(h,v,significant,sign)
%% function ContextSign.m
% �ú������ڻ�ȡ������������Ϣ
% h��ˮƽ�ھӵ���Ҫ����Ϣ
% v����ֱ�ھӵ���Ҫ����Ϣ
% significant��������λ�õ���Ҫ����Ϣ
% sign��������λ�õķ�����Ϣ
% y������ķ���������ֵ
% ho�����ÿ�ַ��������Ķ�Ӧ��ˮƽ�ھӷ�����Ϣ
% ver�����ÿ�ַ��������Ķ�Ӧ�Ĵ�ֱ�ھӷ�����Ϣ
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